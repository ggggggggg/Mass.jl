
module Mass
include("MicrocalFiles.jl")
include("H5Flow.jl")
include("Summarize.jl")
using .Summarize, .MicrocalFiles, .H5Flow
export microcal_open


println("yay imported Mass!")
end # module
