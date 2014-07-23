module LJH

export LJHFile

# LJH file header information
immutable LJHHeader
    filename         ::String
    nPresamples      ::Int64
    nSamples         ::Int64
    timebase         ::Float64
    timestampOffset  ::Float64
    date             ::String
    headerSize       ::Int64
    channum          ::Uint16
end

# LJH file abstraction

type LJHFile
    name             ::String        # filename
    str              ::IOStream      # IOStream to read from LJH file
    header           ::LJHHeader     # LJH file header data
    nrec             ::Int64         # number of (pulse) records in file
    dt               ::Float64       # sample spacing (microseconds)
    npre             ::Int64         # nPresample
    nsamp            ::Int64         # number of sample per record
    reclength        ::Int64         # record length (bytes) including timestamp
    channum          ::Uint16        # channel number  
    function LJHFile(name::ASCIIString)
        hd = readLJHHeader(name)
        channum = hd.channum
        dt = hd.timebase
        pre = hd.nPresamples
        tot = hd.nSamples
        str = open(name)
        datalen = stat(name).size - hd.headerSize
        reclen = 2*(3+tot)
        # assert((datalen%reclen)==0)
        nrec = div(datalen,reclen)
        seek(str,hd.headerSize)
        new(name, str, hd, nrec, dt, pre, tot, reclen, channum)
    end
end
function update_num_records(f::LJHFile)
    datalen = stat(f.name).size - f.header.headerSize
    f.nrec = div(datalen,ljh.reclength)
end


type LJHSlice{T<:AbstractArray}
    ljhfile::LJHFile
    slice::T
    function LJHSlice(ljhfile, slice)
        maximum(slice)<=ljhfile.nrec || error("$(maximum(slice)) is greater than nrec=$(ljhfile.nrec) in $ljhfile")
        new(ljhfile, slice)
    end
end
LJHSlice{T<:AbstractArray}(ljhfile::LJHFile, slice::T) = LJHSlice{T}(ljhfile, slice)


function Base.show(io::IO, f::LJHFile)
    print(io, "LJHFile channel $(f.channum)\n")   
    print(io, f.name*"\n")
    print(io, "$(f.nrec) records with $(f.npre) presamples and $(f.nsamp) samples each\n")
end



# Get pertinent information from LJH file header and return it as LJHHeader
function readLJHHeader(filename::String)
    str=open(filename)
    labels={"base"   =>"Timebase:",
            "date"   =>"Date:",
            "date1"  =>"File First Record Time:",
            "end"    =>"#End of Header",
            "offset" =>"Timestamp offset (s):",
            "pre"    =>"Presamples: ",
            "tot"    =>"Total Samples: ",
            "channum"=>"Channel: "}
    nlines=0
    maxnlines=100
    date = "unknown" # If header standard for date labels changes, we don't want a hard error

    # Read channel # from the file name, then update that result from the header, if it exists.
    channum = uint16(-1)
    m = match(r"_chan\d+", filename)
    channum = uint16(m.match[6:end])

    while nlines<maxnlines
        line=readline(str)
        nlines+=1
        if beginswith(line,labels["end"])
            headerSize = position(str)
            close(str)
            return(LJHHeader(filename,nPresamples,nSamples,
                             timebase,timestampOffset,date,headerSize,channum))
        elseif beginswith(line,labels["base"])
            timebase = float64(line[1+length(labels["base"]):end])
        elseif beginswith(line,labels["date"]) # Old LJH files
            date = line[7:end-2]
        elseif beginswith(line,labels["date1"])# Newer LJH files
            date = line[25:end-2]
        elseif beginswith(line,labels["channum"])# Newer LJH files
            channum = uint16(line[10:end])
        elseif beginswith(line,labels["offset"])
            timestampOffset = float64(line[1+length(labels["offset"]):end])
        elseif beginswith(line,labels["pre"])
            nPresamples = int64(line[1+length(labels["pre"]):end])
        elseif beginswith(line,labels["tot"])
            nSamples = int64(line[1+length(labels["tot"]):end])
        end
    end
    error("read_LJH_header: where's '$(labels["end"])' ?")
end



# Read the next nrec records and for each return time and samples 
# (error if eof occurs or insufficient space in data)
function fileRecords(f::LJHFile, nrec::Integer,
                     times::Vector{Uint64}, data::Matrix{Uint16})
    #assert(nrec <= min(length(times),size(data,2)) && size(data,1)==f.nsamp)
    for i=1:nrec
        times[i] = recordTime(read(f.str, Uint8, 6))
        data[:,i] = read(f.str, Uint16, f.nsamp)
    end
end

LJHRewind(f::LJHFile) = seek(f.str, f.header.headerSize)


# Read specific record numbers and for each return time and samples;
# (error if eof occurs or insufficient space in data)
function fileRecords(f::LJHFile, recIndices::Vector{Int},
                     times::Vector{Uint64}, data::Matrix{Uint16})
    for i = 1:length(recIndices)
        seekto(f,recIndices[i])
        times[i] = recordTime(read(f.str, Uint8, 6))
        data[:,i] = read(f.str, Uint16, f.nsamp)
    end
end

# support for ljhfile[1:7] syntax
seekto(f::LJHFile, i::Int) = seek(f.str,f.header.headerSize+(i-1)*(2*f.nsamp+6))
Base.getindex(f::LJHFile,indexes::AbstractVector)=LJHSlice(f, indexes)
function Base.getindex(f::LJHFile,index::Int)
    seekto(f, index)
    pop!(f)
end
function Base.pop!(f::LJHFile)
    timestamp =  recordTime(read(f.str, Uint8, 6))
    data = read(f.str, Uint16, f.nsamp)
    data, timestamp
end

Base.size(f::LJHFile) = (f.nrec,)
Base.length(f::LJHFile) = f.nrec
Base.endof(f::LJHFile) = f.nrec

# access as iterator
Base.start(f::LJHFile) = (seekto(f,1);1)
Base.next(f::LJHFile,j) = pop!(f),j+1
Base.done(f::LJHFile,j) = j==length(f)+1
Base.length(f::LJHFile) = f.nrec


Base.start{T}(f::LJHSlice{T}) = (j=start(f.slice);seekto(f.ljhfile,j);j)
function Base.next{T<:UnitRange}(f::LJHSlice{T}, j) 
    n,r=next(f.slice,j)
    pop!(f.ljhfile),r
end
function Base.next{T}(f::LJHSlice{T},j)
    n,r=next(f.slice,j)
    f.ljhfile[n],r
end
Base.done{T}(f::LJHSlice{T}, j) = done(f.slice,j)
Base.length{T}(f::LJHSlice{T}) = length(f.slice)
Base.endof{T}(f::LJHSlice{T}) = length(f.slice)

# From LJH file, return all data samples as single vector
function fileData(filename::String)
    ljh = LJHFile(filename)
    [d for (d,t) in ljh]
    close(ljh.str)
    data
end

fileData(ljh::LJHFile) = [d for (d,t) in ljh]
        



# Time in microseconds, given six-byte pulse record header (LJH version 2.1.0)
function recordTime(header::Vector{Uint8})
    frac = uint64(header[1])
    ms = uint64(header[3]) |
         (uint64(header[4])<<8) |
         (uint64(header[5])<<16) |
         (uint64(header[6])<<24)
    return ms*1000 + frac*4
end

# Six-byte pulse record header (LJH version 2.1.0), given time in microseconds
recordHeader(t::Uint64) = recordHeader(t, Array(Uint8,6))

# Allocation-free version
function recordHeader(t::Uint64, header::Vector{Uint8})
    frac = div(t%1000,4)
    ms = uint32(div(t,1000))
    header[1] = uint8(frac)
    header[2] = uint8(0)
    header[3] = uint8(0xFF & ms)
    header[4] = uint8(0xFF & (ms >> 8))
    header[5] = uint8(0xFF & (ms >> 16))
    header[6] = uint8(0xFF & (ms >> 24))
    return header
end

# writing ljh files
function writeLJHHeader(filename::String,dt,npre,nsamp)
    f = open(filename, "w")
    writeLJHHeader(f,dt,npre,nsamp)
    close(f)
end
function writeLJHHeader(io::IO, dt, npre, nsamp)
    write(io,
"#LJH Memorial File Format
Save File Format Version: 2.0.0
Software Version: Fake LJH file converted from ROOT
Software Driver Version: n/a
Date: %(asctime)s GMT
Acquisition Mode: 0
Digitized Word Size in bytes: 2
Location: LANL, presumably
Cryostat: Unknown
Thermometer: Unknown
Temperature (mK): 100.0000
Bridge range: 20000
Magnetic field (mGauss): 100.0000
Detector:
Sample:
Excitation/Source:
Operator: Unknown
SYSTEM DESCRIPTION OF THIS FILE:
USER DESCRIPTION OF THIS FILE:
#End of description
Number of Digitizers: 1
Number of Active Channels: 1
Timestamp offset (s): 1304449182.876200
Digitizer: 1
Description: CS1450-1 1M ver 1.16
Master: Yes
Bits: 16
Effective Bits: 0
Anti-alias low-pass cutoff frequency (Hz): 0.000
Timebase: $(dt)
Number of samples per point: 1
Presamples: $(npre)
Total Samples: $(nsamp)
Trigger (V): 250.000000
Tigger Hysteresis: 0
Trigger Slope: +
Trigger Coupling: DC
Trigger Impedance: 1 MOhm
Trigger Source: CH A
Trigger Mode: 0 Normal
Trigger Time out: 351321
Use discrimination: No
Channel: 1.0
Description: A (Voltage)
Range: 0.500000
Offset: -0.000122
Coupling: DC
Impedance: 1 Ohms
Inverted: No
Preamp gain: 1.000000
Discrimination level (%%): 1.000000
#End of Header\n"
    )
end
function writeLJHData(filename::String, a...)
    f=open(filename, "a")
    writeLJHData(f,a...)
    close(f)
end
function writeLJHData(io::IO,traces::Array{Uint16,2}, times::Vector{Uint64})
    for j = 1:length(times)
        writeLJHData(io, traces[:,j], times[j])
    end
end
function writeLJHData(io::IO, trace::Vector{Uint16}, time::Uint64)
    [write(io,t) for t in recordHeader(time)] # write 6 Uint8s that represent the time
    write(io, trace)
end

end # end module

