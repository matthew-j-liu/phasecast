#=
Stateless MAGEMin equilibrium runner for the Track B corpus.

Reads an input CSV where each row is one (sample_id, T, P, 11 oxides) point and
writes one JSONL record per point. No T/P grid is built here: the caller samples
(P, H2O) and expands the cooling T-ladder (see the Python driver). This script
just minimizes whatever (P, T, X) the input provides and dumps the assemblage.

Input CSV columns (required):
  sample_id, T_celsius, P_kbar, sio2, al2o3, cao, mgo, feo, fe2o3,
  k2o, na2o, tio2, cr2o3, h2o_tot
Extra columns pass through unread.

Output JSONL (one line per input row):
  {"sid", "P", "T", "status", "G",
   "phases": [{"name", "mode_wt", "comp_wt": {oxide: wt%}}],   # SS + PP, in out.ph order
   "ss":     [{"name", "emNames": [...], "emFrac_wt": [...]}]} # SS phases only

out.ph concatenates solution phases (n_SS, aligned to SS_vec) then pure phases
(n_PP, aligned to PP_vec); ph_frac_wt is aligned to out.ph. Both SS and PP carry
Comp_wt; only SS carry endmembers. MAGEMin may repeat a phase name (two fsp or
cpx compositions coexisting) -- duplicates are preserved as separate list
entries; do NOT key phases by name here.

Usage:
  julia --project=magemin -t auto magemin/run_magemin.jl --input IN.csv --output OUT.jsonl [--database ig] [--quiet]
=#

using MAGEMin_C
using CSV
using DataFrames
using JSON
using ArgParse

const OXIDES = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3",
                "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]
const CSV_OXIDES = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
                    "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]


function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input";    help = "Input CSV";  required = true
        "--output";   help = "Output JSONL"; required = true
        "--database"; help = "Thermodynamic database (ig = igneous)"; default = "ig"
        "--quiet";    help = "Suppress the progress bar"; action = :store_true
    end
    return parse_args(s)
end


function build_vectors(df::DataFrame)
    N = nrow(df)
    P = Vector{Float64}(undef, N)
    T = Vector{Float64}(undef, N)
    X = Vector{Vector{Float64}}(undef, N)
    sid = Vector{Int}(undef, N)
    for i in 1:N
        P[i] = df[i, :P_kbar]
        T[i] = df[i, :T_celsius]
        X[i] = [Float64(df[i, c]) for c in CSV_OXIDES]
        sid[i] = df[i, :sample_id]
    end
    return P, T, X, sid
end


"""One JSONL-ready Dict for a single MAGEMin output object."""
function extract_record(out, sid, P, T)
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
    return Dict("sid" => sid, "P" => P, "T" => T,
                "status" => status, "G" => G,
                "phases" => phases, "ss" => ss)
end


function main()
    args = parse_cli()
    df = CSV.read(args["input"], DataFrame)
    println("loaded $(nrow(df)) points from $(args["input"])")

    P, T, X, sid = build_vectors(df)
    N = length(P)

    data = Initialize_MAGEMin(args["database"], verbose=false)
    t0 = time()
    res = multi_point_minimization(P, T, data; X=X, Xoxides=OXIDES,
                                   sys_in="wt", progressbar=!args["quiet"])
    el = time() - t0
    println("  $(round(el, digits=1))s ($(round(1000*el/max(N,1), digits=1)) ms/point)")

    n_ok = 0
    open(args["output"], "w") do io
        for i in 1:N
            local rec
            try
                rec = extract_record(res[i], sid[i], P[i], T[i])
                if get(rec, "status", -1) == 0
                    n_ok += 1
                end
            catch
                rec = Dict("sid" => sid[i], "P" => P[i], "T" => T[i],
                           "status" => -1, "G" => NaN,
                           "phases" => [], "ss" => [])
            end
            println(io, JSON.json(rec))
        end
    end
    Finalize_MAGEMin(data)
    println("  $n_ok / $N converged; wrote $(args["output"])")
end

main()
