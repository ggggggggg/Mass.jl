module MicrocalFiles
include("LJH.jl")
using .LJH
export microcal_open, hdf5_name_from_ljh_name, LJHFile

# the idea is that all microcal files should support the same interfaces, so you
# can open any of them with microcal_open
# f = microcal_open(filename)
# then f[j] should return (data, timestamp) where data is the pulse record and 
# and timestamp is the time the pulse was triggered
# and [(d,t) for (d,t) in f] should iterate across all pulses in the file
function microcal_open(s::String)
	rest, ext = splitext(s)
	lowercase(ext) == ".ljh" && return LJHFile(s)
	lowercase(ext) == ".noi" && return LJHFile(s)
	error("$ext is not a supported file type")
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


end 