module H5Helper
using HDF5
export g_create_or_open, d_update, a_update
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

end