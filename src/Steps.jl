include("H5Helper.jl")
using H5Helper, Logging, Mass
Logging.configure(level=DEBUG)

immutable Step
    func::String
    a_ins::(String...) #attribute inputs
    d_ins::(String...) #dataset inputs
    a_outs::(String...) #attribute outputs
    d_outs::(String...) #dataset outputs
    Step(func,a,b,c,d) = new(func, tupleize(a), tupleize(b), tupleize(c), tupleize(d))
end
==(a::Step, b::Step) = all([getfield(a,name)==getfield(b,name) for name in names(a)])
tupleize(x::String) = (x,)
tupleize(x) = tuple(x...)
input_lengths(h5grp, s::Step) = [length(h5grp[name]) for name in s.d_ins]
output_lengths(h5grp, s::Step) = [H5Helper.exists(h5grp, name) ? length(h5grp[name]) : 0 for name in s.d_outs]
range(h5grp, s::Step) = maximum(input_lengths(h5grp, s))+1:minimum(output_lengths(h5grp,s))
a_args(h5grp, s::Step) = [a_read(h5grp,name) for name in s.a_ins]
d_args(h5grp, s::Step, r::UnitRange) = [h5grp[name][r] for name in s.d_ins]
args(h5grp, s::Step, r::UnitRange) = tuple(a_args(h5grp, s)..., d_args(h5grp, s, r)...)
calc_outs(h5grp, s::Step, r::UnitRange) = eval(parse(s.func))(args(h5grp, s, r)...)
function place_outs(h5grp, s::Step, r::UnitRange, outs) 
    for j in 1:length(s.a_outs) a_update(h5grp, s.a_outs[j], outs[j]) end
    for j in length(s.a_outs)+1:length(s.a_outs) d_extend(h5grp, d_outs[j], outs[j], r) end
end
function dostep(h5grp, s::Step)
    r = range(h5grp,s)
    info(name(h5grp), " ",r, " ", s)
    place_outs(h5grp, s, r, calc_outs(h5grp, s, r))
end
h5step_write(h5grp, s::Step) = for name in names(s) a_require(h5grp, "$name", repr(getfield(s,name))) end
h5step_read(h5grp)=Step([eval(parse(a_read(h5grp, "$name"))) for name in names(Step)]...)
h5step_add(h5grp, s::Step, n::Integer) = h5step_write(g_require(g_require(h5grp, "steps"),"$n"), s)
function h5steps(h5grp)
    g=g_require(h5grp, "steps")
    nums = sort([int(name) for name in names(g)])
    [h5step_read(g["$n"]) for n in nums]
end
update!(h5grp::HDF5Group) = [(println(s);dostep(h5grp, s)) for s in h5steps(h5grp)]

ljh=microcal_open("/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh")
h5 = h5open(hdf5_name_from_ljh_name(ljh.name),"w")
close(h5)
@time o2=Mass.Summarize.summarize_flow(ljh)


h5 = h5open(hdf5_name_from_ljh_name(ljh.name),"r+")
g=g_require(h5,"chan1")
a_update(g, "pretrigger_mean_correction", [2000.0, 0.1])

function ptm_correction(params, ptm, ph)
    ptm_offset, slope = params
    ph += (ptm.-ptm_offset).*ph
    return (ph,)
end
s = Step("ptm_correction", "pretrigger_mean_correction", ["summary/pretrig_mean","summary/pulse_rms"], (), "pulse_rms_dc")
h5step_add(g,s, 10)
update!(g)

h5step_write(g,s)
s2 = h5step_read(g)
@assert(s2==s)