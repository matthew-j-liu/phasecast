using MAGEMin_C, CSV, DataFrames, JSON
df = CSV.read(ARGS[1], DataFrame)
OX = ["SiO2","Al2O3","CaO","MgO","FeO","Fe2O3","K2O","Na2O","TiO2","Cr2O3","H2O"]
COL = ["sio2","al2o3","cao","mgo","feo","fe2o3","k2o","na2o","tio2","cr2o3","h2o_tot"]
P = Float64.(df.P_kbar); T = Float64.(df.T_celsius)
X = [ [Float64(df[i,c]) for c in COL] for i in 1:nrow(df) ]
data = Initialize_MAGEMin("ig", verbose=false)
res = multi_point_minimization(P, T, data; X=X, Xoxides=OX, sys_in="wt", progressbar=false)
open(ARGS[2],"w") do io
  for i in 1:nrow(df)
    o = res[i]; oxn = String.(o.oxides)
    phases = Dict{String,Any}()
    for (nm, ss, fr) in zip(o.ph, o.SS_vec, o.ph_frac_wt)
      phases[nm] = Dict("mode"=>fr, "comp_wt"=>Dict(zip(oxn, round.(Float64.(ss.Comp_wt),digits=4))))
    end
    rec = Dict("sid"=>df[i,:sample_id], "G"=>(try o.G_system catch; NaN end), "phases"=>phases)
    println(io, JSON.json(rec))
  end
end
Finalize_MAGEMin(data)
