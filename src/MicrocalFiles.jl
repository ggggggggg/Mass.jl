module MicrocalFiles
include("LJH.jl")
using .LJH
export microcal_open, hdf5_name_from_ljh, channel, record_nsamples, pretrig_nsamples, timebase, filenames, LJHFile, LJHGroup

# the idea is that all microcal files should support the same interfaces, so you
# can open any of them with microcal_open
# f = microcal_open(filename)
# then f[j] should return (data, timestamp) where data is the pulse record and 
# and timestamp is the time the pulse was triggered
# and [(d,t) for (d,t) in f] should iterate across all pulses in the file
# I'm trying to only export LJHGroup, and allow you to use it for a single file 
# or group of files
function microcal_open(s::String)
	rest, ext = splitext(s)
	lowercase(ext) == ".ljh" && return LJHGroup(s)
	lowercase(ext) == ".noi" && return LJHGroup(s)
	error("$ext is not a supported file type")
end
microcal_open(s) = LJHGroup(s) # for lists of files, I didn't bother to make it able to handle other types of files, shouldn't be hard to add

# Given an LJH file name, return the HDF5 name
# Generally, /x/y/z/data_taken_chan1.ljh becomes /x/y/z/data_taken_mass.hdf5
function ljhbasename(ljhname::String)
    bname,suffix = splitext(basename(ljhname))
    m = match(r"_chan\d+", bname)
    m == nothing ? bname : bname[1:m.offset-1]
end
hdf5_name_from_ljh(ljhgroup::LJHGroup) = hdf5_name_from_ljh(fnames(ljhgroup)...)
function hdf5_name_from_ljh(ljhnames::String...)
	dir = dirname(ljhnames[1])
	fname = prod([ljhbasename(f) for f in ljhnames])
	joinpath(dir,hdf5_name_from_ljh(fname))
end
hdf5_name_from_ljh(ljhname::String) = joinpath(dirname(ljhname),ljhbasename(ljhname))*"_mass.hdf5"
hdf5_name_from_ljh(ljh::LJHFile) = hdf5_name_from_ljh(ljh.name)

end # module