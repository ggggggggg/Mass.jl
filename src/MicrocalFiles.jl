include("LJH.jl")
module MicrocalFiles
using LJH
export microcal_open

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


end 