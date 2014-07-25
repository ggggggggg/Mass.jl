include("H5Helper.jl")
using H5Helper, Logging
Logging.configure(level=DEBUG)


function dostep(h5grp::HDF5Group, func::Symbol, a_inputs, d_inputs, a_outputs, d_outputs)
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

g["d"]
