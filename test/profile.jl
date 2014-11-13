using Mass, ArrayViews

@everywhere begin 
function ptm_correction(params, ptm, ph)
    ptm_offset, slope = params
    ph += (ptm.-ptm_offset).*ph
    return ph
end
ptm_correction_step = Step(ptm_correction, "pretrigger_mean_correction", ["pretrig_mean","pulse_rms"], (), "pulse_rms_dc")
type Calibration
	features::Vector{ASCIIString}
	energies::Vector{Float64}
	estimates::Vector{Float64}
end

count_bools(a,b) = int(sum(a)+b)
count_good_step = Step(count_bools, "number_of_good_pulses","selections/good", "number_of_good_pulses",[])

function calibrate(pulse_rms)
	println("Calibration!!!!**!!")
	@show (length(pulse_rms))
	(Calibration(["Zero","MnKAlpha"], [0, 5898], [0,median(pulse_rms)]),)
end
calibrate_step = SelectedStep(calibrate, "good", [], "pulse_rms", "calibration/pulse_rms")

function apply(cal::Calibration, pulse_rms)
	energy = pulse_rms.*(cal.energies[end]/cal.estimates[end])
	return energy
end
apply_calibration_step = Step(apply, "calibration/pulse_rms", "pulse_rms", [], "energy")


end # everywhere

tnow, tlast = time(), time()
tstart = time()
println("starting loop!")
pulse_steps_done = 1 # initializat with any nonzero value
max_chans = 1
#while true
	#ljhname, ljhopen = MicrocalFiles.matter_writing_status()
	ljhname, ljhopen = "/Volumes/Drobo/exafs_data/20141008_brown_ferrioxalate_straw_2mm_emission/20141008_brown_ferrioxalate_straw_2mm_emission_chan1.ljh", false
	jldname = hdf5_name_from_ljh(ljhname)
	jldname = "/Volumes/Drobo/exafs_data/20141008_brown_ferrioxalate_straw_2mm_emission/20141008_brown_ferrioxalate_straw_2mm_emission_mass_julia.hdf5"
	jldopen(jldname,"w") do jld
		chans_to_init = 1:2:2*max_chans
		println("initialized file $ljhname channels $chans_to_init")
		cg = init_channels(jld, ljhname, chans_to_init)
		h5step_add(jld, summarize_step)
	end
	tnow, tlast = time(), tnow
	sleeptime = 1-(tnow-tlast)
	sleeptime > 0 && sleep(sleeptime)
	jld = jldopen(jldname,"r+")
	println("update w summarize")
	@time update!(jld, 10000)
	close(jld)

	ljhname, ljhopen = "/Volumes/Drobo/exafs_data/20141008_brown_ferrioxalate_straw_2mm_emission/20141008_brown_ferrioxalate_straw_2mm_emission_chan1.ljh", false

ljhgroup = microcal_open(ljhname)
# #@time out = summarizep(ljhgroup, 1:10000)
#  using PyPlot
#      Nsamp = record_nsamples(ljhgroup)
#     Npre = pretrig_nsamples(ljhgroup)+2
#     Npost = Nsamp-Npre
# pre_x = [1-div(Npre,2):Npre-div(Npre,2)]
# post_x = [Npre+2:Nsamp]
# data,timestamp = ljhgroup[5]
# pre_poly = polyfit(pre_x, view(data[1:Npre]),1)    
# post_poly = polyfit(post_x, view(data[post_x]), 8)

# plot(data,".")
# plot(pre_x+div(Npre,2), polyval(pre_x, pre_poly))
# plot(post_x, polyval(post_x, post_poly))


# close(ljhgroup)


# N=1000
# a = 500*(rand(N)+rand())

# s=0.0
# s2=0.0
# s22=0.0
# for j = 1:N
# 	s+=a[j]
# 	s2+=a[j]*a[j]
# 	s22+=(a[j]-mean(a))^2
# end

# # std_loop = sqrt(s2/N+s*s/(N*N)-2*s*s/(N*N*N))
# std_loop = sqrt(s2/N-s*s/(N*N))
# std_loop2 = sqrt(s22/N)
# std_normal = std(a)

# println("std_loop $std_loop, std_normal $std_normal, diff $(std_loop-std_normal)")
# println("std_loop $std_loop2, std_normal $std_normal, diff $(std_loop2-std_normal)")
# println("std_loop $std_loop, std_normal $std_loop2, diff $(std_loop-std_loop2)")


# # Estimate the derivative (units of arbs / sample) for a pulse record or other timeseries.
# # This version uses the default kernel of [-2,-1,0,1,2]/10.0
# #
# max_timeseries_deriv!(deriv, pulserecord, reject_spikes::Bool) =
#     max_timeseries_deriv!(deriv, pulserecord, convert(Vector{eltype(deriv)},[.2 : -.1 : -.2]), reject_spikes)


# # Post-peak derivative computed using Savitzky-Golay filter of order 3
# # and fitting 1 point before...3 points after.
# #
# max_timeseries_deriv_SG!(deriv, pulserecord, reject_spikes::Bool) =
#     max_timeseries_deriv!(deriv, pulserecord, [-0.11905, .30952, .28572, -.02381, -.45238],
#                             reject_spikes)

# # Estimate the derivative (units of arbs / sample) for a pulse record or other timeseries.
# # Caller pre-allocates the full derivative array, which is available as deriv.
# # Returns the maximum value of the derivative.
# # The kernel should be a short *convolution* (not correlation) kernel to be convolved
# # against the input pulserecord.
# # If reject_spikes is true, then the max value at sample i is changed to equal the minimum
# # of the values at (i-2, i, i+2). Note that this test only makes sense for kernels of length
# # 5 (or less), because only there can it be guaranteed insensitive to unit-length spikes of
# # arbitrary amplitude.
# #
# function max_timeseries_deriv!{T}(
#         deriv::Vector{T},       # Modified! Pre-allocate an array of sufficient length
#         pulserecord, # The pulse record (presumably starting at the pulse peak)
#         kernel::Vector{T},      # The convolution kernel that estimates derivatives
#         reject_spikes::Bool  # Whether to employ the spike-rejection test
#         )
#     N = length(pulserecord)
#     Nk = length(kernel)
#     @assert length(deriv) >= N+1-Nk
#     if Nk > N
#         return 0.0
#     end
#     if Nk+4 > N
#         reject_spikes = false
#     end

#     for i=1:N-Nk+1
#         deriv[i] = 0 # int, but auto converted?
#         for j=1:Nk

#             deriv[i] += pulserecord[i+Nk-j]*kernel[j] #float
#         end
#     end
#     for i=N-Nk+2:length(deriv)
#         deriv[i]=deriv[N-Nk+1]
#     end
#     if reject_spikes
#         for i=3:N-Nk-2
#             if deriv[i] > deriv[i+2]
#                 deriv[i] = deriv[i+2]
#             end
#             if deriv[i] > deriv[i-2]
#                 deriv[i] = deriv[i-2]
#             end
#         end
#     end
#     maximum(deriv)
# end

# a = linspace(0,1,100)
# deriv = zeros(a)

# m = max_timeseries_deriv!(deriv, a, false)