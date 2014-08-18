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
immutable Step <: AbstractStep
    func::String
    o_ins::(ASCIIString...) # other inputs
    p_ins::(ASCIIString...) # per pulse inputs
    o_outs::(ASCIIString...) # other outputs
    p_outs::(ASCIIString...) # per pulse outputs
    Step(func,a,b,c,d) = new(func, tupleize(a), tupleize(b), tupleize(c), tupleize(d))
end
Step(func::Symbol,a,b,c,d) = Step(string(func),a,b,c,d)
Step(func::Function,a,b,c,d) = Step("$func",a,b,c,d)
==(a::Step, b::Step) = all([getfield(a,name)==getfield(b,name) for name in names(a)])
tupleize(x::String) = (x,)
tupleize(x) = tuple(x...)
input_lengths(jlgrp, s::Step) = [input_length(jlgrp[name]) for name in s.p_ins]
input_length(d::JldDataset) = size(d) == () ? read(d) : size(d)[end]
# input lengths is a vector of the lengths of all the input datasets and "npulses" (with a default of 0 if it doesn't exist)
output_lengths(jlgrp, s::Step) = length(s.p_outs) == 0 ? [0] : [[exists(jlgrp, name) ? size(jlgrp[name])[end] : 0 for name in s.p_outs]]
range(jlgrp, s::Step) = minimum(output_lengths(jlgrp, s))+1:minimum(input_lengths(jlgrp,s))
o_args(jlgrp, s::Step) = [read(jlgrp,name) for name in s.o_ins]
p_args(jlgrp, s::Step, r::UnitRange) = [size(jlgrp[name])==() ? read(jlgrp[name]) : jlgrp[name][r] for name in s.p_ins]
args(jlgrp, s::Step, r::UnitRange) = tuple(o_args(jlgrp, s)..., p_args(jlgrp, s, r)...)
calc_outs(jlgrp, s::Step, r::UnitRange) = getfield(Main,symbol(s.func))(args(jlgrp, s, r)...)
function place_outs(jlgrp, s::Step, r::UnitRange, outs)
    assert(length(outs) == length(s.o_outs)+length(s.p_outs))
    for j in 1:length(s.o_outs) 
        update!(jlgrp, s.o_outs[j], outs[j]) end
    isempty(r) && return #dont try to place dataset outs with empty range
    for j in 1:length(s.p_outs) 
        d_extend(jlgrp, s.p_outs[j], outs[j+length(s.o_outs)], r) end
end
dostep(jlgrp::Union(JldFile, JldGroup), s::Step) = dostep(jlgrp, s, range(jlgrp,s))
function dostep(jlgrp::Union(JldFile, JldGroup), s::Step, r::UnitRange)
    starttime = tic()
    outs = calc_outs(jlgrp, s, r)
    elapsed = (tic()-starttime)*1e-9
    println(name(jlgrp), " ",r, " ",elapsed," s, ", s)
    place_outs(jlgrp, s, r, outs)
end
function dostep(jlgrp::Union(JldFile, JldGroup), s::Step, max_step_size::Int)
    inputs_exist(jlgrp,s) || (println(name(jlgrp), " inputs don't exist, so skipping ",s);return)
    r = range(jlgrp, s)
    length(r)>max_step_size && (r = first(r):max_step_size-first(r)%max_step_size+first(r))
    dostep(jlgrp, s, r)
end
h5step_add(jlgrp::Union(JldFile, JldGroup), s::AbstractStep, n::Integer) = jlgrp["steps/$n"] = s
function h5step_add(jlgrp::Union(JldFile, JldGroup), s::AbstractStep)
    nums = h5stepnumbers(jlgrp)
    n = isempty(nums) ? 10 : 10*div(last(nums),10)+10
    h5step_add(jlgrp, s, n)
end
function h5step_del(jlgrp::Union(JldFile, JldGroup), s::AbstractStep)
    nums, steps = h5stepnumbers(jlgrp), h5steps(jlgrp)
    println(steps)
    del_nums = nums[steps.==s]
    length(del_nums) >0 || error("$s is not a step in $jlgrp")
    println(del_nums)
    for n in del_nums delete!(jlgrp,"steps/$n") end
    nums
end
function h5stepnumbers(jlgrp::Union(JldFile, JldGroup))
    exists(jlgrp, "steps") || return Int[]
    nums = sort([int(name) for name in names(jlgrp["steps"])])
end
function h5steps(jlgrp::Union(JldFile, JldGroup))
    nums = h5stepnumbers(jlgrp)
    AbstractStep[read(jlgrp["steps"]["$n"]) for n in nums] # no check for existing because non empty nums requires existence
end
update!(jlgrp::JldGroup) = [dostep(jlgrp, s, typemax(Int)) for s in h5steps(jlgrp)]
update!(jlgrp::JldGroup, max_step_size::Int) = [dostep(jlgrp, s, max_step_size) for s in h5steps(jlgrp)]

immutable ThresholdStep{T<:AbstractStep} <: AbstractStep
    watched_dset::ASCIIString
    thresholdlength::Real
    step::T
end
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
function dostep(jlgrp::Union(JldFile, JldGroup), s::ThresholdStep, max_step_size::Int)
    outputs_exist(jlgrp, s.step) && return 
    dsetlength = size(jlgrp[s.watched_dset])[end]
    dsetlength < s.thresholdlength && return
    dostep(jlgrp, s.step, max_step_size)
end

immutable NothingStep <: AbstractStep
end
function dostep(jlgrp::Union(JldFile, JldGroup), s::NothingStep, max_step_size::Int)
    debug("doing NothingStep")
end
immutable RangeStep <: AbstractStep
    s::Step
end
RangeStep(a...) = RangeStep(Step(a...))
function dostep(jlgrp::Union(JldFile, JldGroup), s::RangeStep, r::UnitRange)
    starttime = tic()
    outs = calc_outs(jlgrp, s, r)
    elapsed = (tic()-starttime)*1e-9
    println(name(jlgrp), " ",r, " ",elapsed," s, ", s)
    place_outs(jlgrp, s.s, r, outs)
end
function dostep(jlgrp::Union(JldFile, JldGroup), s::RangeStep, max_step_size::Int)
    inputs_exist(jlgrp,s.s) || (println(name(jlgrp), " inputs don't exist, so skipping ",s);return)
    r = range(jlgrp, s.s)
    length(r)>max_step_size && (r = first(r):max_step_size-first(r)%max_step_size+first(r))
    dostep(jlgrp, s, r)
end
calc_outs(jlgrp, s::RangeStep, r::UnitRange) = getfield(Main,symbol(s.s.func))(r, args(jlgrp, s.s, r)...)


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
       Step, AbstractStep, ThresholdStep, RangeStep,
       update!, h5steps, h5step_add

end # endmodule

