#=
Fractional crystallization runner for the generator-reach exploration: the
production runner's minimization, stepped down a cooling ladder with every
phase but the melt removed after each step.

Reads the same input CSV as magemin/run_magemin.jl plus a `path_id` column:
rows sharing a path_id are one cooling path (one bulk at one P and H2O), run in
descending T. Each path starts from the 11-oxide bulk of its hottest row with
the system mass at 1; after a converged step the next step's bulk is the melt's
own composition (its oxides including the excess oxygen `O`, so iron speciation
follows oxygen mass balance, as MAGEMinCalc does) and the system mass is scaled
by the melt's weight fraction. A failed step (status != 0, G non-finite or >= 0)
is written and skipped, the bulk kept. A path ends when no melt is stable, when
the melt falls below --melt-floor of the starting mass, or at its last row;
rows below the end are not run and not written.

Output JSONL: one record per attempted step with run_magemin.jl's fields plus
"path_id", "mass" (system fraction of the starting mass entering the step; a
phase's mass on the starting basis is mass x mode_wt) and "end" ("ladder",
"melt_out", "melt_floor" on the path's last record, null elsewhere).

Usage:
  julia --project=magemin -t auto explorations/generator_reach/run_fractional.jl \
      --input IN.csv --output OUT.jsonl [--database ig] [--melt-floor 1e-4] [--quiet]
=#

using MAGEMin_C
using CSV
using DataFrames
using JSON
using ArgParse
using Base.Threads

const OXIDES = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3",
                "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]
const CSV_OXIDES = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
                    "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]


function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input";      help = "Input CSV";  required = true
        "--output";     help = "Output JSONL"; required = true
        "--database";   help = "Thermodynamic database (ig = igneous)"; default = "ig"
        "--melt-floor"; help = "Stop a path when the melt falls below this fraction of the starting mass"; arg_type = Float64; default = 1e-4
        "--quiet";      help = "Suppress progress"; action = :store_true
    end
    return parse_args(s)
end


function extract_record(out, sid, P, T, path_id, mass)
    oxn = String.(out.oxides)
    n_SS = out.n_SS

    phases = Vector{Any}()
    ss = Vector{Any}()
    for (i, name) in enumerate(out.ph)
        comp = i <= n_SS ? out.SS_vec[i].Comp_wt : out.PP_vec[i - n_SS].Comp_wt
        push!(phases, Dict(
            "name" => name,
            "mode_wt" => out.ph_frac_wt[i],
            "comp_wt" => Dict(zip(oxn, round.(Float64.(comp), digits=4))),
        ))
        if i <= n_SS
            ssd = out.SS_vec[i]
            if !isempty(getfield(ssd, :emNames))
                push!(ss, Dict(
                    "name" => name,
                    "emNames" => ssd.emNames,
                    "emFrac_wt" => round.(Float64.(ssd.emFrac_wt), digits=4),
                ))
            end
        end
    end

    status = try out.status catch; -1 end
    G = try Float64(out.G_system) catch; NaN end
    return Dict("sid" => sid, "P" => P, "T" => T, "path_id" => path_id, "mass" => mass,
                "status" => status, "G" => G, "end" => nothing,
                "phases" => phases, "ss" => ss)
end


function failed_line(sid, P, T, path_id, mass, last)
    JSON.json(Dict("sid" => sid, "P" => P, "T" => T, "path_id" => path_id, "mass" => mass,
                   "status" => -1, "G" => nothing, "end" => last ? "ladder" : nothing,
                   "phases" => [], "ss" => []))
end


"""Run one path; returns its JSONL lines in step order."""
function run_path(rows::SubDataFrame, data, db, melt_floor)
    id = Threads.threadid()
    gv = data.gv[id]
    z_b = data.z_b[id]
    DB = data.DB[id]
    splx_data = data.splx_data[id]

    first = rows[1, :]
    X0 = [Float64(first[c]) for c in CSV_OXIDES]
    gv = define_bulk_rock(gv, X0, OXIDES, "wt", db)
    mass = 1.0
    path_id = Int(first[:path_id])

    lines = String[]
    for k in 1:nrow(rows)
        r = rows[k, :]
        sid, P, T = Int(r[:sample_id]), Float64(r[:P_kbar]), Float64(r[:T_celsius])
        local rec
        try
            out = point_wise_minimization(P, T, gv, z_b, DB, splx_data)
            rec = extract_record(out, sid, P, T, path_id, mass)
            if rec["status"] == 0 && isfinite(rec["G"]) && rec["G"] < 0
                iliq = findfirst(==("liq"), out.ph)
                if isnothing(iliq)
                    rec["end"] = "melt_out"
                elseif mass * out.ph_frac_wt[iliq] < melt_floor
                    rec["end"] = "melt_floor"
                else
                    gv = define_bulk_rock(gv, Float64.(out.SS_vec[iliq].Comp_wt),
                                          String.(out.oxides), "wt", db)
                    mass *= out.ph_frac_wt[iliq]
                end
            end
            if isnothing(rec["end"]) && k == nrow(rows)
                rec["end"] = "ladder"
            end
            push!(lines, JSON.json(rec))
        catch
            # JSON.jl refuses NaN, and a non-converged point can carry one; a
            # bad record must not kill the path, and the bulk stays as it was
            push!(lines, failed_line(sid, P, T, path_id, mass, k == nrow(rows)))
            continue
        end
        if !isnothing(rec["end"])
            break
        end
    end
    return lines
end


function main()
    args = parse_cli()
    df = CSV.read(args["input"], DataFrame)
    df = sort(df, [:path_id, order(:T_celsius, rev=true)])
    paths = collect(groupby(df, :path_id; sort=true))
    println("loaded $(nrow(df)) points on $(length(paths)) paths from $(args["input"])")

    db = args["database"]
    data = Initialize_MAGEMin(db, verbose=false)
    out_lines = Vector{Vector{String}}(undef, length(paths))
    t0 = time()
    @threads :static for i in eachindex(paths)
        out_lines[i] = run_path(paths[i], data, db, args["melt-floor"])
    end
    el = time() - t0
    n_steps = sum(length.(out_lines))
    println("  $(round(el, digits=1))s ($(round(1000*el/max(n_steps,1), digits=1)) ms/step, $n_steps steps run)")

    n_ok = 0
    open(args["output"], "w") do io
        for lines in out_lines, line in lines
            println(io, line)
            n_ok += occursin("\"status\":0", line) ? 1 : 0
        end
    end
    Finalize_MAGEMin(data)
    println("  $n_ok / $n_steps converged; wrote $(args["output"])")
end

main()
