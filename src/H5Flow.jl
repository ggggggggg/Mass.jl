module H5Flow
using HDF5, JLD, Logging
import JLD: JldGroup, JldFile, JldDataset
import HDF5: a_read,==, read

# there are three relevant things in an HDF5 file
# g_* deals with groups, d_* deals with datasets, a_* deals with atttributes
# groups contain other object, they are like folders
# roughly we plan to put each channel in it's own group, and use a few more groups for organization
# attributes are for small things, like single numbers, strings, or small sets of numbers
# datasets are for large things, for th purposes of mass this means anything where you have one item per record

# Create a new or open an existing group within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.

function g_require(parent::Union(JldFile,JldGroup), name::ASCIIString)
    exists(parent, name) ? (return parent[name]) : g_create(parent, name)
end
# Create a new or update an existing dataset within an HDF5 object
# extends the dataset if required
function d_extend(parent::HDF5Group, name::ASCIIString, value::Vector, range::UnitRange)
	d = d_require(parent, name, value)
	set_dims!(parent[name], (maximum(range),))
	d[range] = value
	d
end
d_extend(parent::HDF5Group, name::ASCIIString, value::Vector) = d_extend(parent, name, value, endof(parent[name])+1:endof(parent[name])+length(value))
d_update(parent::HDF5Group, name::ASCIIString, value::Vector) = d_extend(parent, name, value, 1:endof(value))
function d_require(parent::HDF5Group, name, value::Vector,chunksize = 10000)
	dims = ((1,), (-1,)) # create a minimum size dataset, zero isn't allowed
	exists(parent,name) ? parent[name] : d_create(parent, name, eltype(value), dims, "chunk", (chunksize,))
end
# Create a new or update an existing attribute within an HDF5 object
# a_require will create the attribute if it doesn't exist, or assert that the existing attribute is equal to value
function a_require(parent::HDF5Group,name::ASCIIString,value)
    if exists(attrs(parent), name)
    	a_read(parent, name) == value ? (return value) : error("new value $value != existing value $(a_read(parent,name)) for attr $parent[$name]")
	end
    attrs(parent)[name] = value	
end
# a_update will create or replace and existing attribute with value
function a_update(parent::HDF5Group,name::ASCIIString,value)
    if exists(attrs(parent), name)
        old_value = a_read(parent, name)
    	old_value == value && (return value)
    	a_delete(parent, name)
        info("in $parent updating $name from $old_value to $value")
	end
    attrs(parent)[name] = value
end
# read an attribute, but if it doesn't exist, return default_value
function a_read(parent::HDF5Group, name::String, default_value)
	exists(attrs(parent),name) ? a_read(parent, name) : default_value
end
allnames(parent::Union(HDF5Group, HDF5File)) = (names(parent), names(attrs(parent)))
function read(parent::Union(JldGroup, JldFile), name::ASCIIString, default_value)
    exists(parent, name) ? read(parent, name) : default_value
end
# foward functions from Jld Group/File to plain 
for f in (:d_extend, :d_update, :d_require, :a_require, :a_update, :allnames, :a_read)
eval(:($f(parent::Union(JldFile, JldGroup), args...) = $f(parent.plain, args...)))
end
function update!(parent::Union(JldFile, JldGroup), name::ASCIIString, value)
    if exists(parent, name)
        old_value = read(parent, name)
        old_value == value && (return value)
        typeof(old_value) == typeof(value) || error("attempted to update $(parent[name]) of type $(typeof(old_value)) with new value of type $(typeof(value))")
        delete!(parent[name])
    end
    parent[name] = value
end


# Given an LJH file name, return the HDF5 name
# Generally, /x/y/z/data_taken_chan1.ljh becomes /x/y/z/data_taken_mass.hdf5
function hdf5_name_from_ljh_name(ljhname::String)
    dir = dirname(ljhname)
    base = basename(ljhname)
    path,suffix = splitext(ljhname)
    m = match(r"_chan\d+", path)
    path = string(path[1:m.offset-1], "_mass.hdf5")
end

abstract AbstractStep
==(a::AbstractStep, b::AbstractStep) = typeof(a)==typeof(b) && all([getfield(a,n)==getfield(b,n) for n in names(a)])
calc_outs(jlgrp, s::AbstractStep, r::UnitRange) = (a=args(jlgrp,s,r);@spawn func(s)(a...))
function dostep(jlgrp::Union(JldFile, JldGroup), s::AbstractStep, r::UnitRange)
    length(r)>0 || (println("$r has length < 1 so skipping $s"); return)
    starttime = tic()
    outsref = calc_outs(jlgrp, s, r)
    elapsed = (tic()-starttime)*1e-9
    println(name(jlgrp), " ",r, " ",elapsed," s, ", s)
    outsref, r
    # place_outs(jlgrp, s, r, outs)
end
function dostep(jlgrp::Union(JldFile, JldGroup), s::AbstractStep, max_step_size::Int)
    inputs_exist(jlgrp,s) || (println(name(jlgrp), " inputs don't exist, so skipping ",s);return)
    r = range(jlgrp, s)
    length(r)>max_step_size && (r = first(r):max_step_size-first(r)%max_step_size+first(r))
    dostep(jlgrp, s, r)
end

immutable Step <: AbstractStep
    func::String
    o_ins::(ASCIIString...) # other inputs
    p_ins::(ASCIIString...) # per pulse inputs
    o_outs::(ASCIIString...) # other outputs
    p_outs::(ASCIIString...) # per pulse outputs
    Step(func,a,b,c,d) = new(func, tupleize(a), tupleize(b), tupleize(c), tupleize(d))
end
Base.show(io::IO, s::Step) = print(io, "Step with function $(s.func), other inputs $(s.o_ins), per pulse inputs $(s.p_ins), other outputs $(s.o_outs), per pulse outputs $(s.p_outs)")
Step(func::Symbol,a,b,c,d) = Step(string(func),a,b,c,d)
Step(func::Function,a,b,c,d) = Step("$func",a,b,c,d)
==(a::Step, b::Step) = all([getfield(a,name)==getfield(b,name) for name in names(a)])
tupleize(x::String) = (x,)
tupleize(x) = tuple(x...)
func(s::Step) = getfield(Main, symbol(s.func))
input_lengths(jlgrp, s::Step) = [input_length(jlgrp[name]) for name in s.p_ins]
input_length(d::JldDataset) = size(d) == () ? read(d) : size(d)[end]
function output_lengths(jlgrp, s::Step)
    length(s.p_outs) == 0 && (return [0])
    [[!exists(jlgrp, name) ? 0 : (size(jlgrp[name])==() ? read(jlgrp[name]) : size(jlgrp[name])[end]) for name in s.p_outs]]
end
o_args(jlgrp, s::Step) = [read(jlgrp,name) for name in s.o_ins]
p_args(jlgrp, s::Step, r::UnitRange) = [size(jlgrp[name])==() ? read(jlgrp[name]) : jlgrp[name][r] for name in s.p_ins]
args(jlgrp, s::Step, r::UnitRange) = tuple(o_args(jlgrp, s)..., p_args(jlgrp, s, r)...)
function outputs_exist(jlgrp, s::Step)
    p_outs = [exists(jlgrp, name) for name in s.p_outs]
    o_outs = [exists(jlgrp, name) for name in s.o_outs]
    all(p_outs) && all(o_outs)
end
function inputs_exist(jlgrp, s::Step)
    p_ins = [exists(jlgrp, name) for name in s.p_ins]
    o_ins = [exists(jlgrp, name) for name in s.o_ins]
    all(p_ins) && all(o_ins)
end
p_inputs(s::Step) = s.p_ins
o_inputs(s::Step) = s.o_ins
p_outputs(s::Step) = s.p_outs
o_outputs(s::Step) = s.o_outs
funcname(s::Step) = s.func

function place_outs(jlgrp, s::Step, r::UnitRange, outs::NTuple)
    assert(length(outs) == length(s.o_outs)+length(s.p_outs))
    for j in 1:length(s.o_outs) 
        update!(jlgrp, s.o_outs[j], outs[j]) end
    isempty(r) && return #dont try to place dataset outs with empty range
    for j in 1:length(s.p_outs) 
        out = outs[j+length(s.o_outs)]
        if typeof(out)<:Int
            update!(jlgrp, s.p_outs[j], out)
        else
            d_extend(jlgrp, s.p_outs[j], out, r) 
        end
    end
end

h5step_add(jlgrp::JldGroup, s::AbstractStep, n::Integer) = jlgrp["steps/$n"] = s
function h5step_add(jlgrp::JldGroup, s::AbstractStep)
    nums = h5stepnumbers(jlgrp)
    n = isempty(nums) ? 10 : 10*div(last(nums),10)+10
    h5step_add(jlgrp, s, n)
end
function h5step_del(jlgrp::JldGroup, s::AbstractStep)
    nums, steps = h5stepnumbers(jlgrp), h5steps(jlgrp)
    println(steps)
    del_nums = nums[steps.==s]
    length(del_nums) >0 || error("$s is not a step in $jlgrp")
    println(del_nums)
    for n in del_nums delete!(jlgrp,"steps/$n") end
    nums
end
function h5stepnumbers(jlgrp::JldGroup)
    exists(jlgrp, "steps") || return Int[]
    nums = sort([int(name) for name in names(jlgrp["steps"])])
end
function h5steps(jlgrp::JldGroup)
    nums = h5stepnumbers(jlgrp)
    AbstractStep[read(jlgrp["steps"]["$n"]) for n in nums] # no check for existing because non empty nums requires existence
end
function h5step_add(jld::JldFile, s::AbstractStep)
    for c in chans(jld)
        h5step_add(c,s)
    end
end

function update!(jlgrp::JldGroup, max_step_size::Int)
    pulse_steps = 0 # track how much work was done
    for s in h5steps(jlgrp)
        step_result = dostep(jlgrp, s, max_step_size)
        if step_result != nothing
            outsref, r = step_result
            place_jlgrp(outs, s, r, fetch(outsref))
	    pulse_steps += length(r)
        end
    end
    pulse_steps
end
update!(jlgrp::JldGroup) = update(jlgrp, typemax(Int))
chans(jld::Union(JldFile, JldGroup)) = filter!(s->beginswith(name(s), "/chan"), [g for g in jld])
update!(jld::JldFile) = update!(jld, typemax(Int))
function update!(jld::JldFile, max_step_size::Int)
    pulse_steps_done = 0
    stepnumbers = IntSet()
    channels = chans(jld)
    for c in channels, n in h5stepnumbers(c)
        push!(stepnumbers, n)
    end
    
    for n in stepnumbers
	pulses_done = 0
	tstart = time()
        q = Dict()
        for c in channels
            if exists(c,"steps/$n")
                step = read(c["steps/$n"])
                q[c] = dostep(c, step, max_step_size)
            end
        end
        for c in channels
            step_result = get(q,c,nothing)
            if step_result != nothing
                outsref,r = step_result
                step = read(c["steps/$n"])
        		outs = fetch(outsref)
        		typeof(outs) <: Exception && error("$outs\n the above exception occured on $step on $c")
                place_outs(c, step, r, outs)
        		pulse_steps_done += length(r)
        		pulses_done += length(r)
            end
        end
	tend = time()
	step = read(first(channels)["steps/$n"]) # assume all channels have same steps with same numbers
	println("stepnumber $n, $pulses_done processed in $(tend-tstart) s, $(pulses_done/(tend-tstart)) pulses/s on $(length(channels)) channels step $step")

    end
    pulse_steps_done
end




### Forward functions for AbstractStep to Step ###
input_lengths(jlgrp, s::AbstractStep) = input_lengths(jlgrp, s.s)
output_lengths(jlgrp, s::AbstractStep) = output_lengths(jlgrp, s.s)
o_args(jlgrp, s::AbstractStep) = o_args(jlgrp, s.s)
p_args(jlgrp, s::AbstractStep, r::UnitRange) = p_args(jlgrp, s.s, r)
args(jlgrp, s::AbstractStep, r::UnitRange) = tuple(o_args(jlgrp, s)..., p_args(jlgrp, s, r)...)
func(s::AbstractStep) = func(s.s)
place_outs(jlgrp, s::AbstractStep, r::UnitRange, outs::NTuple) = place_outs(jlgrp, s.s, r, outs)
place_outs(jlgrp, s::AbstractStep, r::UnitRange, outs) = place_outs(jlgrp, s, r, tuple(outs)) # convert non-tuples
dostep(jlgrp::Union(JldFile, JldGroup), s::AbstractStep) = dostep(jlgrp, s, range(jlgrp,s))
inputs_exist(jlgrp, s::AbstractStep) = inputs_exist(jlgrp, s.s)
outputs_exist(jlgrp, s::AbstractStep) = outputs_exist(jlgrp, s.s)
p_inputs(s::AbstractStep) = p_inputs(s.s)
o_inputs(s::AbstractStep) = o_inputs(s.s)
p_outputs(s::AbstractStep) = p_outputs(s.s)
o_outputs(s::AbstractStep) = o_outputs(s.s)
funcname(s::AbstractStep) = funcname(s.s)
function range(jlgrp, s::AbstractStep)
    try
        minimum(output_lengths(jlgrp, s))+1:minimum(input_lengths(jlgrp,s))
    catch
        0:0
    end
end
# range(jlgrp, s::AbstractStep) = minimum(output_lengths(jlgrp, s))+1:minimum(input_lengths(jlgrp,s))
### ThresholdStep waits until a JldDatset either is long enough, or its value is large enough ###
### Then it preforms its contained step if and only if the outputs from the contained step don't exist ###
### It is for things like calibration that need a certain amount of data, but only need to be done once ###
immutable ThresholdStep{T<:AbstractStep} <: AbstractStep
    watched_dset::ASCIIString
    thresholdlength::Real
    s::T
end
function dostep(jlgrp::Union(JldFile, JldGroup), s::ThresholdStep, max_step_size::Int)
    inputs_exist(jlgrp,s) || (println(name(jlgrp), " inputs don't exist, so skipping ",s);return)
    outputs_exist(jlgrp, s) && return 
    dsetlength = minimum(input_lengths(jlgrp, s))
    dsetlength < s.thresholdlength && return
    dostep(jlgrp, s.s, max_step_size)
end


### NothingStep does nothing, mostly for testing purposes ###
immutable NothingStep <: AbstractStep
end
function dostep(jlgrp::Union(JldFile, JldGroup), s::NothingStep, max_step_size::Int)
    debug("doing NothingStep")
end
### Range Step ### 
### Range Step is a good exampleof how to create a specific type of step by overloading a few functions ###
### Range Step is used by summarize because it passes a range arugment to the calculating function ###
### thats how summarize knows which pulses to look at ###
immutable RangeStep <: AbstractStep
    s::Step
end
RangeStep(a...) = RangeStep(Step(a...))
calc_outs(jlgrp, s::RangeStep, r::UnitRange) = (a=args(jlgrp, s.s, r);@spawn getfield(Main,symbol(s.s.func))(r, a...))
function dostep(jlgrp::Union(JldFile, JldGroup), s::RangeStep, r::UnitRange)
    starttime = tic()
    outsref = calc_outs(jlgrp, s, r)
    elapsed = (tic()-starttime)*1e-9
    #println(name(jlgrp), " ",r, " ",elapsed," s, ", s)
    outsref, r
    # place_outs(jlgrp, s, r, outs)
end
### Selection helper functions ###
selection_g_name = "selections"
selection_names(g::JldGroup) = convert(Vector{ASCIIString}, names(g[selection_g_name]))
selection_lengths(g::JldGroup, names::Vector{ASCIIString}) = [length(g[selection_g_name][n]) for n in names]
selection_lengths(g::JldGroup) = selection_lengths(g,selection_names(g))
selection_extend(g::JldGroup, name::ASCIIString, v::Vector{Uint8}, a...) = d_extend(g, name, v, a...)
selection_extend(g::JldGroup, name::ASCIIString, v::Vector{Bool}, a...) = d_extend(g, name, reinterpret(Uint8,v), a...)
selection_extend(g::JldGroup, name::ASCIIString, v::BitArray{1}, a...) = selection_extend(g, name, convert(Vector{Bool}, v), a...)
selection_read(g::JldGroup, name::ASCIIString, r::UnitRange) = reinterpret(Bool, g[name][r])
selection_read(g::JldGroup, name) = reinterpret(Bool, read(g[name]))
function select_lims(lims, v)
    l,h = minmax(lims...)
    println(lims, " ", l, " ", h)
    out = l .< v .< h
    println("select_lims selected $(sum(out)) of $(length(out))")
    out
end
### SelectingStep writes Vector{Uint8} HDF5 arrays from functions that return Vector{Bool} or BitVector ###
### SelectingStep may only have outputs of type BitVector and Vector{Bool} ###
immutable SelectingStep <: AbstractStep
    s::Step
end
SelectingStep(a::ASCIIString,b::ASCIIString) = SelectingStep(Step(select_lims,a,b,(),joinpath(selection_g_name,b)))
SelectingStepGood(a::Vector{ASCIIString}) = SelectingStep(Step(&,(),[joinpath(selection_g_name,n) for n in a],(),joinpath(selection_g_name,"good")))
function place_outs(jlgrp, s::SelectingStep, r::UnitRange, outs::NTuple)
    assert(length(outs) == length(s.s.o_outs)+length(s.s.p_outs))
    for o in outs
        typeof(o) <: Union(BitVector, Vector{Bool}, Vector{Uint8}) || error("SelectingStep may only have outputs of type BitVector, Vector{Bool}, Vector{Uint8}, not $(typeof(o))")
    end    
    for j in 1:length(s.s.o_outs) 
        update!(jlgrp, s.s.o_outs[j], outs[j]) end
    isempty(r) && return #dont try to place dataset outs with empty range
    for j in 1:length(s.s.p_outs) 
        selection_extend(jlgrp, s.s.p_outs[j], outs[j+length(s.s.o_outs)], r) end     
    for j in 1:length(s.s.p_outs) 
        counter_name = s.s.p_outs[j]*"_count"
        previous_count = read(jlgrp, counter_name,0)
        new_count = previous_count + sum(outs[j+length(s.s.o_outs)])
        update!(jlgrp, counter_name, new_count)
    end
end
### Selected Step uses only certain entries in the p_ins. For example if you want the pulse_rms value from only the "good" pulses ###
immutable SelectedStep <: AbstractStep
    selections::(ASCIIString...)
    s::Step
    num_seen_name::ASCIIString
    function SelectedStep(s_ins, s)
        length(s.p_outs)==0 || error("SelectedStep p_outs must be a single integer")
        s_ins = tupleize([beginswith(s_in, selection_g_name)?s_in:joinpath(selection_g_name,s_in) for s_in in s_ins])
        new(tupleize(s_ins),s,selection_g_name*string(hash(s)))
    end
end
SelectedStep(f,a,b,c,d,e) = SelectedStep(tupleize(a),Step(f,b,c,d,e))
function selection(jlgrp, s::SelectedStep, r::UnitRange)
    # for now this just takes the first selection listed
    # in the future I'd like to support something like "good and (pumped or unpumped)"
    # which would selection all the pulses that are in selection good and in selection pumped or unpumped
    length(s.selections)>1 && warn("currently SelectedStep only supports one selection, so it will only use the first of $(s.selections)")
    selected = selection_read(jlgrp, s.selections[1], r)
    println("$(name(jlgrp)) from selections $(s.selections) $(sum(selected)) selected pulses of $(length(selected)) possible pulses")
    selected
end
function p_args(jlgrp, s::SelectedStep, r::UnitRange)
    s_selection = selection(jlgrp, s, r)
    [jlgrp[name][r][s_selection] for name in s.s.p_ins]
end
function place_outs(jlgrp, s::SelectedStep, r::UnitRange, outs::NTuple)
    place_outs(jlgrp, s.s, r, outs)
    num_seen = read(jlgrp, s.num_seen_name, 0) + length(r)
    update!(jlgrp, s.num_seen_name, num_seen)
end
output_lengths(jlgrp, s::SelectedStep) = [[read(jlgrp, s.num_seen_name, 0)]]

## One time step
immutable OneTimeStep <: AbstractStep
    s::Step
end
function dostep(jlgrp::Union(JldFile, JldGroup), s::OneTimeStep, max_step_size::Int)
    inputs_exist(jlgrp,s) || (println(name(jlgrp), " inputs don't exist, so skipping ",s);return)
    outputs_exist(jlgrp,s) && (println(name(jlgrp), " outputs exist, so skipping ",s);return)
    r=0:0
    outsref = calc_outs(jlgrp, s, r)
    outsref,r
end

function pythonize(jlgrp::JldGroup, o_ins, a_outs)
    # non per pulse "o" datasets get written as attrs
    # for compatability with python mass
    for (o,a) in zip(o_ins, a_outs)
        println(o," ",a)
        value = read(jlgrp, o)
        delete!(jlgrp, o)
        a_require(jlgrp, a, value)
    end
end

export g_require, # group stuff
       d_update, d_extend, d_require, #dataset stuff
       a_update, a_require, a_read, # attribute stuff
       hdf5_name_from_ljh_name, jldopen, allnames,
       close, JldGroup, JldFile, name, attrs, names,
       Step, AbstractStep, ThresholdStep, RangeStep, SelectingStep, SelectingStepGood, select_lims, select_and,
       SelectedStep,
       update!, h5steps, h5step_add
end # endmodule

