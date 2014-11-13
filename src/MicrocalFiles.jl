module MicrocalFiles
include("LJH.jl")
using .LJH
export microcal_open, hdf5_name_from_ljh, channel, record_nsamples, pretrig_nsamples, frametime, 
	   filenames, lengths, LJHFile, LJHGroup, column, row, num_columns, num_rows, 
	   ljhsplit, ljhchannel, ljhfnames, ljhallchannels, ljhall, matter_writing_status

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
function microcal_open(f::Function, args...)
    io = microcal_open(args...)
    try
	f(io)
    finally
	close(io)
    end
end
microcal_open(s) = LJHGroup(s) # for lists of files, I didn't bother to make it able to handle other types of files, shouldn't be hard to add

# Given an LJH file name, return the HDF5 name
# Generally, /x/y/z/data_taken_chan1.ljh becomes /x/y/z/data_taken_mass.hdf5
function ljhsplit(ljhname::String)
    if isdir(ljhname)
        dname = ljhname
        bname = last(split(dname,'/'))
        return dname, bname
    end
    bname,suffix = splitext(basename(ljhname))
    m = match(r"_chan\d+", bname)
    dirname(ljhname), m == nothing ? bname : bname[1:m.offset-1]
end
function ljhchannel(ljhname::String)
    m = match(r"_chan(\d+)", ljhname)
    m == nothing ? -1 : int(m.captures[1])
end
function ljhfnames(ljhname::String, chans)
    dname, bname = ljhsplit(ljhname)
    [joinpath(dname, "$(bname)_chan$c.ljh") for c in chans]
end
function ljhallchannels(ljhname::String)
    dname, bname = ljhsplit(ljhname)
    potential_ljh = filter!(s->(beginswith(s,bname)), readdir(dname))   
    channels = filter!(x->x>=0,[ljhchannel(p) for p in filter!(s->beginswith(s,bname),readdir(dname))])
    sort!(unique(channels))
end
ljhall(ljhname::String) = ljhfnames(ljhname, ljhallchannels(ljhname))
hdf5_name_from_ljh(ljhgroup::LJHGroup) = hdf5_name_from_ljh(filenames(ljhgroup)...)
function hdf5_name_from_ljh(ljhnames::String...)
	dname, bname = ljhsplit(ljhnames[1])
	fname = prod([ljhsplit(f)[2] for f in ljhnames])
	joinpath(dname,hdf5_name_from_ljh(fname))
end
hdf5_name_from_ljh(ljhname::String) = joinpath(ljhsplit(ljhname)...)*"_mass.hdf5"
hdf5_name_from_ljh(ljh::LJHFile) = hdf5_name_from_ljh(ljh.name)

# returns (String, Bool) representing (filename, currently_open?)
function matter_writing_status()
       sentinel_dir = joinpath(expanduser("~"),".daq")
       isdir(sentinel_dir) || error("$(sentinel_dir) must be a directory")
       sentinel_file_path = joinpath(sentinel_dir,"latest_ljh_pulse.cur")
       isfile(sentinel_file_path) || error("$(sentinel_file) must be a file")
       sentinel_file = open(sentinel_file_path,"r")
       lines = map(chomp,collect(eachline(sentinel_file)))
       close(sentinel_file)
       lines[1], length(lines)==1 # the sentinel file has a second line that says closed when it has closed a file
end

end # module
