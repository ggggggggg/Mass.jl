module H5Flow
using HDF5, Logging

# there are three relevant things in an HDF5 file
# g_* deals with groups, d_* deals with datasets, a_* deals with atttributes
# groups contain other object, they are like folders
# roughly we plan to put each channel in it's own group, and use a few more groups for organization
# attributes are for small things, like single numbers, strings, or small sets of numbers
# datasets are for large things, for th purposes of mass this means anything where you have one item per record

# Create a new or open an existing group within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.
function g_require(parent::Union(HDF5File,HDF5Group), name::String)
    exists(parent, name) ? (return parent[name]) : g_create(parent, name)
end
# Create a new or update an existing dataset within an HDF5 object
# extends the dataset if required
function d_extend(parent::HDF5Group, name::String, value::Vector, range::UnitRange)
	d = d_require(parent, name, value)
	set_dims!(parent[name], (maximum(range),))
	d[range] = value
	d
end
d_extend(parent::HDF5Group, name::String, value::Vector) = d_extend(parent, name, value, endof(parent[name])+1:endof(parent[name])+length(value))
d_update(parent::HDF5Group, name::String, value::Vector) = d_extend(parent, name, value, 1:endof(value))
function d_require(parent::HDF5Group, name, value::Vector,chunksize = 10000)
	dims = ((1,), (-1,)) # create a minimum size dataset, zero isn't allowed
	exists(parent,name) ? parent[name] : d_create(parent, name, eltype(value), dims, "chunk", (chunksize,))
end
# Create a new or update an existing attribute within an HDF5 object
# a_require will create the attribute if it doesn't exist, or assert that the existing attribute is equal to value
function a_require(parent::HDF5Group,name::String,value)
    if exists(attrs(parent), name)
    	a_read(parent, name) == value ? (return value) : error("new value $value != existing value $(a_read(parent,name)) for attr $parent[$name]")
	end
    attrs(parent)[name] = value	
end
# a_update will create or replace and existing attribute with value
function a_update(parent::HDF5Group,name::String,value)
    if exists(attrs(parent), name)
    	a_read(parent, name) == value && (return value)
    	a_delete(parent, name)
	end
    attrs(parent)[name] = value
end
# read an attribute, but if it doesn't exist, return default_value
function HDF5.a_read(parent::HDF5Group, name::String, default_value)
	exists(attrs(parent),name) ? a_read(parent, name) : default_value
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

immutable Step
    func::Function
    a_ins::(UTF8String...) #attribute inputs
    d_ins::(UTF8String...) #dataset inputs
    a_outs::(UTF8String...) #attribute outputs
    d_outs::(UTF8String...) #dataset outputs
    Step(func,a,b,c,d) = new(func, tupleize(a), tupleize(b), tupleize(c), tupleize(d))
end
Step(func::String,a,b,c,d,m::Module=Main) = Step(symbol(func),a,b,c,d,m)
Step(func::Symbol,a,b,c,d,m::Module=Main) = Step(getfield(m, func),a,b,c,d)
==(a::Step, b::Step) = all([getfield(a,name)==getfield(b,name) for name in names(a)])
tupleize(x::String) = (x,)
tupleize(x) = tuple(x...)
input_lengths(h5grp, s::Step) = [length(h5grp[name]) for name in s.d_ins]
output_lengths(h5grp, s::Step) = [exists(h5grp, name) ? length(h5grp[name]) : 0 for name in s.d_outs]
range(h5grp, s::Step) = maximum(input_lengths(h5grp, s))+1:minimum(output_lengths(h5grp,s))
a_args(h5grp, s::Step) = [a_read(h5grp,name) for name in s.a_ins]
d_args(h5grp, s::Step, r::UnitRange) = [h5grp[name][r] for name in s.d_ins]
args(h5grp, s::Step, r::UnitRange) = tuple(a_args(h5grp, s)..., d_args(h5grp, s, r)...)
calc_outs(h5grp, s::Step, r::UnitRange) = s.func(args(h5grp, s, r)...)
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
function h5step_read(h5grp,m::Module=Main)
    data = [tupleize([convert(UTF8String,m.match) for m in collect(eachmatch(r"[/0-9a-zA-Z_]+", a_read(h5grp, "$name")))]) for name in names(Step)]
    Step(getfield(m, symbol(data[1][1])), data[2:end]...)   
end
h5step_add(h5grp, s::Step, n::Integer) = h5step_write(g_require(g_require(h5grp, "steps"),"$n"), s)
function h5steps(h5grp)
    exists(h5grp, "steps") || return Step[]
    g=g_require(h5grp, "steps")
    nums = sort([int(name) for name in names(g)])
    Step[h5step_read(g["$n"]) for n in nums]
end
update!(h5grp::HDF5Group) = [(println(s);dostep(h5grp, s)) for s in h5steps(h5grp)]

export g_require, # group stuff
       d_update, d_extend, d_require, #dataset stuff
       a_update, a_require, a_read, # attribute stuff
       hdf5_name_from_ljh_name, h5open, 
       close, HDF5Group, HDF5File, name, attrs, names,
       Step, update!, h5steps, h5step_add

end # endmodule

