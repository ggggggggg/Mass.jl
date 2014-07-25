include("H5Helper.jl")
using H5Helper, Logging, Mass
Logging.configure(level=DEBUG)


function dostep(h5grp::HDF5Group, func::Symbol, a_inputs::(String...), d_inputs::(String...), a_outputs::(String...), d_outputs::(String...))
    println(d_inputs)
    input_lengths = [length(h5grp[name]) for name in d_inputs]
    output_lengths = [H5Helper.exists(h5grp, name) ? length(h5grp[name]) : 0 for name in d_outputs]
    s,f = maximum(output_lengths)+1,minimum(input_lengths)
    info("dostep ", func, " on $(name(h5grp))[$s:$f] $a_inputs and $d_inputs going to $a_outputs and $d_outputs")
    s<f || return
    a_args = [a_read(h5grp,name) for name in a_inputs]
    d_args = [h5grp[name][s:f] for name in d_inputs]

    rets = eval(func)(a_args..., d_args...)
    j=1
    for name in a_outputs
        @show a_update(h5grp, name, rets[j])
        @show j+=1
    end 
    for name in d_outputs
        @show d_extend(h5grp, name, rets[j], s:f)
        @show j+=1
    end
    return
end
dostep(h5grp, func, a,b,c,d) = dostep(h5grp, func, tupleize(a), tupleize(b), tupleize(c), tupleize(d))
tupleize(x::String) = (x,)
tupleize(x) = tuple(x...)
h5 = h5open("test.h5","w")
g = g_require(h5, "chan1")
g["a"] = [1:100]
g["b"] = ones(100)
attrs(g)["c"] = [4,5,6]

function f(c,a,b)
	# @assert([4,5,6]==c)
	# @assert([1:100]==a)
	# @assert(ones(100)==b)
	(a.*b.+sum(c),)
end
@show dostep(g,:f, ["c"], ["a", "b"], [], ["d"])

function ptm_correction(params, ptm, ph)
    ptm_offset, slope = params
    ph += (ptm-ptm_offset).*ph
    return (ph,)
end

ljh=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")
@time o2=Mass.Summarize.summarize_flow(ljh)
h5 = h5open(hdf5_name_from_ljh_name(ljh.name),"r+")
g=h5["chan1"]

a_update(g, "pretrigger_mean_correction", [2000.0, 0.1])
@show dostep(g, :ptm_correction, "pretrigger_mean_correction", ["summary/pretrig_mean","summary/pulse_rms"], (), "pulse_rms_dc")

