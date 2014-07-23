module H5Helper
using HDF5
export g_create_or_open, d_update, a_update, hdf5_name_from_ljh_name
# Create a new or open an existing group within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.
function g_create_or_open(parent::Union(HDF5File,HDF5Group), name::String)
    exists(parent, name) && return parent[name]
    g_create(parent, name)
end



# Create a new or update an existing dataset within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.
function d_update(parent::HDF5Group, name::String, value)
    if exists(parent, name)
        parent[name][:] = value
    else
        parent[name] = value
    end
end

# Create a new or update an existing dataset within an HDF5 object
# If you can figure out a native syntax that handles both cases,
# then we'd prefer to use it.
function a_update(parent::HDF5Group,name::String,value)
    if exists(attrs(parent), name)
        # Don't do anything if the existing attribute is already correct
        a_read(parent, name) == value && return value
        a_delete(parent, name)
    end
    attrs(parent)[name] = value
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
end # endmodule

