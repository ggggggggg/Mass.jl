module H5Flow
using HDF5, JLD, Logging
import JLD: JldGroup, JldFile
import HDF5: a_read

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

# foward most function from Jld Group/File to plain 
for f in (:d_extend, :d_update, :d_require, :a_require, :a_update, :allnames, :a_read)
eval(:($f(parent::Union(JldFile, JldGroup), args...) = $f(parent.plain, args...)))
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
immutable Step <: AbstractStep
    func::String
    a_ins::(ASCIIString...) #attribute inputs
    d_ins::(ASCIIString...) #dataset inputs
    a_outs::(ASCIIString...) #attribute outputs
    d_outs::(ASCIIString...) #dataset outputs
    Step(func,a,b,c,d) = new(func, tupleize(a), tupleize(b), tupleize(c), tupleize(d))
end
Step(func::Symbol,a,b,c,d) = Step(string(func),a,b,c,d)
Step(func::Function,a,b,c,d) = Step("$func",a,b,c,d)
==(a::Step, b::Step) = all([getfield(a,name)==getfield(b,name) for name in names(a)])
tupleize(x::String) = (x,)
tupleize(x) = tuple(x...)
input_lengths(h5grp, s::Step) = [[size(h5grp[name])[end] for name in s.d_ins], a_read(h5grp, "npulses", 0)]
# input lengths is a vector of the lengths of all the input datasets and "npulses" (with a default of 0 if it doesn't exist)
output_lengths(h5grp, s::Step) = length(s.d_outs) == 0 ? [1] : [[exists(h5grp, name) ? size(h5grp[name])[end] : 0 for name in s.d_outs]]
range(h5grp, s::Step) = minimum(output_lengths(h5grp, s))+1:minimum(input_lengths(h5grp,s))
a_args(h5grp, s::Step) = [a_read(h5grp,name) for name in s.a_ins]
d_args(h5grp, s::Step, r::UnitRange) = [h5grp[name][r] for name in s.d_ins]
args(h5grp, s::Step, r::UnitRange) = tuple(a_args(h5grp, s)..., d_args(h5grp, s, r)...)
calc_outs(h5grp, s::Step, r::UnitRange) = getfield(Main,symbol(s.func))(r, args(h5grp, s, r)...)
function place_outs(h5grp, s::Step, r::UnitRange, outs) 
    assert(length(outs) == length(s.a_outs)+length(s.d_outs))
    for j in 1:length(s.a_outs) 
        a_update(h5grp, s.a_outs[j], outs[j]) end
    isempty(r) && return #dont try to place dataset outs with empty range
    for j in 1:length(s.d_outs) 
        d_extend(h5grp, s.d_outs[j], outs[j+length(s.a_outs)], r) end
end
dostep(h5grp::Union(JldFile, JldGroup), s::Step) = dostep(h5grp, s, range(h5grp,s))
function dostep(h5grp::Union(JldFile, JldGroup), s::Step, r::UnitRange)
    println(s)
    starttime = tic()
    outs = calc_outs(h5grp, s, r)
    elapsed = (tic()-starttime)*1e-9
    println(name(h5grp), " ",r, " ",elapsed," s, ", s)
    place_outs(h5grp, s, r, outs)
end
function dostep(h5grp::Union(JldFile, JldGroup), s::Step, max_step_size::Int)
    r = range(h5grp, s)
    length(r)>max_step_size && (r = first(r):max_step_size-first(r)%max_step_size+first(r))
    dostep(h5grp, s, r)
end
h5step_add(jlgrp::Union(JldFile, JldGroup), s::AbstractStep, n::Integer) = jlgrp["steps/$n"] = s
function h5step_add(jlgrp::Union(JldFile, JldGroup), s::AbstractStep)
    nums = h5stepnumbers(jlgrp)
    n = isempty(nums) ? 10 : 10*div(last(nums),10)+10
    h5step_add(jlgrp, s, n)
end
function h5stepnumbers(jlgrp::Union(JldFile, JldGroup))
    exists(jlgrp, "steps") || return Int[]
    nums = sort([int(name) for name in names(jlgrp["steps"])])
end
function h5steps(jlgrp::Union(JldFile, JldGroup))
    nums = h5stepnumbers(jlgrp)
    AbstractStep[read(jlgrp["steps"]["$n"]) for n in nums] # no check for existing because non empty nums requires existence
end
update!(h5grp::JldGroup) = [dostep(h5grp, s, typemax(Int)) for s in h5steps(h5grp)]
update!(h5grp::JldGroup, max_step_size::Int) = [dostep(h5grp, s, max_step_size) for s in h5steps(h5grp)]

immutable ThresholdStep{T<:AbstractStep} <: AbstractStep
    hashappened::Bool
    thresholdattr::ASCIIString
    thresholdvalue::Real
    step::T
end
ThresholdStep(attr::ASCIIString, value::Real, step::AbstractStep) = ThresholdStep(false, attr, value, step)
ThresholdStep(hashappened::Bool, ts::ThresholdStep) = ThresholdStep(hashappend, ts.thresholdattr, ts.thresholdvalue, ts.step)
function dostep(h5grp::Union(JldFile, JldGroup), s::ThresholdStep, max_step_size::Int)
    s.hashappened && return
    attrvalue = a_read(h5grp, s.thresholdattr, 0)
    attrvalue < s.thresholdvalue && return
    dostep(h5grp, s.step, max_step_size)
    # code to make s.hashappened=true next time we load this step
end

immutable NothingStep <: AbstractStep
end
function dostep(h5grp::Union(JldFile, JldGroup), s::NothingStep, max_step_size::Int)
    debug("doing NothingStep")
end

export g_require, # group stuff
       d_update, d_extend, d_require, #dataset stuff
       a_update, a_require, a_read, # attribute stuff
       hdf5_name_from_ljh_name, jldopen, allnames,
       close, JldGroup, JldFile, name, attrs, names,
       Step, AbstractStep, ThresholdStep,
       update!, h5steps, h5step_add

end # endmodule

