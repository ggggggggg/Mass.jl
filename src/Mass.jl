
module Mass
include("MicrocalFiles.jl")
include("H5Helper.jl")
include("Summarize.jl")
using .Summarize, .MicrocalFiles
export microcal_open


println("yay imported Mass!")
end # module
