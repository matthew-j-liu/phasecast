#=
run_magemin.jl — Stateless MAGEMin runner.

Reads an input CSV where each row is one (sample_id, T, P, 11 oxides) tuple,
calls MAGEMin per row, writes results CSV.

This is intentionally stateless: it does NOT construct a T/P grid. The caller
(02b_sample_tp.py upstream) is responsible for sampling (T, P) per bulk via
rock-class-conditional priors. The Julia script just runs whatever (T, P, X)
the input provides.

Input CSV columns (required):
  sample_id, T_celsius, P_kbar, sio2, al2o3, cao, mgo, feo, fe2o3,
  k2o, na2o, tio2, cr2o3, h2o_tot

Extra columns (fe_provenance, qap_name, etc.) pass through harmlessly — they
are not read.

Output CSV columns:
  sample_id, P_kbar, T_celsius, status, G_system, phases_json, ss_json

Failed points (non-success status) have empty JSON + NaN G_system.

Usage:
  julia --project=. -t auto run_magemin.jl --input INPUT.csv --output OUTPUT.csv [--database ig]
=#

using MAGEMin_C
using CSV
using DataFrames
using JSON
using ArgParse

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input"
            help = "Input CSV with sample_id, T_celsius, P_kbar, and oxide columns"
            required = true
        "--output"
            help = "Output CSV path"
            required = true
        "--database"
            help = "Thermodynamic database (ig = igneous, mp = metapelite)"
            default = "ig"
        "--quiet"
            help = "Suppress MAGEMin progress bar (useful for orchestrator chunks)"
            action = :store_true
    end
    return parse_args(s)
end

# MAGEMin's Xoxides order (mixed case)
const OXIDES = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "Fe2O3",
                "K2O", "Na2O", "TiO2", "Cr2O3", "H2O"]
# Matching CSV column names (lowercase, with h2o_tot for H2O)
const CSV_OXIDES = ["sio2", "al2o3", "cao", "mgo", "feo", "fe2o3",
                    "k2o", "na2o", "tio2", "cr2o3", "h2o_tot"]


"""Build P, T, X vectors directly from input rows (no grid expansion)."""
function build_vectors(df::DataFrame)
    N = nrow(df)
    P_vec = Vector{Float64}(undef, N)
    T_vec = Vector{Float64}(undef, N)
    X_vec = Vector{Vector{Float64}}(undef, N)
    sample_ids = Vector{Int}(undef, N)

    for i in 1:N
        P_vec[i] = df[i, :P_kbar]
        T_vec[i] = df[i, :T_celsius]
        X_vec[i] = [df[i, col] for col in CSV_OXIDES]
        sample_ids[i] = df[i, :sample_id]
    end

    return P_vec, T_vec, X_vec, sample_ids
end


"""Extract phase + SS info from a single MAGEMin output object as JSON strings."""
function extract_result(out)
    status = try out.status catch; -1 end

    phases_json = JSON.json(Dict(
        "ph" => out.ph,
        "ph_frac_wt" => out.ph_frac_wt,
    ))

    ss_dict = Dict{String, Any}()
    for (name, ss) in zip(out.ph, out.SS_vec)
        if !isempty(getfield(ss, :emNames))
            ss_dict[name] = Dict(
                "emNames" => ss.emNames,
                "emFrac_wt" => ss.emFrac_wt,
            )
        end
    end
    ss_json = JSON.json(ss_dict)

    G_sys = try out.G_system catch; NaN end
    return status, G_sys, phases_json, ss_json
end


function main()
    args = parse_cli()
    in_path = args["input"]
    out_path = args["output"]

    println("Loading $in_path")
    comps = CSV.read(in_path, DataFrame)
    println("  $(nrow(comps)) (T, P, composition) tuples loaded")

    P_vec, T_vec, X_vec, sample_ids = build_vectors(comps)
    N_total = length(P_vec)

    println("Initializing MAGEMin (database: $(args["database"]))")
    data = Initialize_MAGEMin(args["database"], verbose=false)

    println("Running multi_point_minimization on $N_total points...")
    t0 = time()
    results = multi_point_minimization(
        P_vec, T_vec, data;
        X = X_vec,
        Xoxides = OXIDES,
        sys_in = "wt",
        progressbar = !args["quiet"],
    )
    elapsed = time() - t0
    ms_per = round(1000 * elapsed / max(N_total, 1), digits=1)
    println("  Done in $(round(elapsed, digits=1))s ($(ms_per) ms/point)")

    println("Extracting results...")
    df_out = DataFrame(
        sample_id = Int[],
        P_kbar = Float64[],
        T_celsius = Float64[],
        status = Int[],
        G_system = Float64[],
        phases_json = String[],
        ss_json = String[],
    )

    n_success = 0
    n_fail = 0
    for i in 1:N_total
        out_i = results[i]
        try
            status, G, ph_json, ss_json = extract_result(out_i)
            push!(df_out, (sample_ids[i], P_vec[i], T_vec[i], status, G, ph_json, ss_json))
            if status == 0
                n_success += 1
            else
                n_fail += 1
            end
        catch e
            push!(df_out, (sample_ids[i], P_vec[i], T_vec[i], -1, NaN, "{}", "{}"))
            n_fail += 1
        end
    end

    pct = round(100 * n_success / max(N_total, 1), digits=1)
    println("  $n_success / $N_total converged ($pct%)")

    Finalize_MAGEMin(data)

    CSV.write(out_path, df_out)
    println("Wrote $out_path")
end

main()
