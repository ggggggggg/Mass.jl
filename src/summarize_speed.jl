type PulseSummaries
    pretrig_mean      ::Vector{Float64}
    pretrig_rms       ::Vector{Float64}
    pulse_average     ::Vector{Float64}
    pulse_rms         ::Vector{Float64}
    rise_time         ::Vector{Float64}
    postpeak_deriv    ::Vector{Float64}#todo
    timestamp         ::Vector{Float64}
    peak_index        ::Vector{Uint16}
    peak_value        ::Vector{Uint16}
    min_value         ::Vector{Uint16}


    function PulseSummaries(n::Integer)
        pretrig_mean = Array(Float64, n)
        pretrig_rms = Array(Float64, n)
        pulse_average = Array(Float64, n)
        pulse_rms = Array(Float64, n)
        rise_time = Array(Float64, n)
        postpeak_deriv = Array(Float64, n)
        timestamp = Array(Float64, n)
        peak_index = Array(Uint16, n)
        peak_value = Array(Uint16, n)
        min_value = Array(Uint16, n)

        new(pretrig_mean, pretrig_rms, pulse_average, pulse_rms, rise_time,
            postpeak_deriv, timestamp, peak_index, peak_value, min_value)
    end
end


# function compute_summary(ljhgroup::LJHGroup, r::UnitRange)
#     summary = PulseSummaries(length(r))
#     Nsamp = record_nsamples(ljhgroup)
#     Npre = pretrig_nsamples(ljhgroup)+2
#     Npost = Nsamp-Npre
#     timebase = frametime(ljhgroup)
#     post_peak_deriv_vect = zeros(Float64, Npost)
#     for (p, (data, timestamp)) in enumerate(ljhgroup[r])
#         # Pretrigger computation first
#         s = s2 = 0.0
#         for j = 1:Npre
#             d = data[j]
#             s += d
#             s2 += d*d
#         end
#         ptm = s/Npre
#         summary.pretrig_mean[p] = ptm
#         summary.pretrig_rms[p] = sqrt(abs(s2/Npre - ptm*ptm))
#         summary.timestamp[p] = timestamp

#         # Now post-trigger calculations
#         s = s2 = 0.0
#         peak_idx = 0
#         peak_val = uint16(0)
#         for j = Npre+1:Nsamp
#             d = data[j]
#             if d > peak_val 
#                 peak_idx, peak_val = j, d
#             end
#             d = d-ptm
#             s += d
#             s2 += d^2
#         end
#         avg = s/Npost

#         posttrig_data = sub(data,Npre+2:endof(data))
#         rise_time::Float64 = estimate_rise_time(posttrig_data, peak_idx-Npre-2,
#                                        peak_val, ptm, timebase)

#         postpeak_data = data[peak_idx+1:end]
#         const reject_spikes=true
#         postpeak_deriv::Float64 = max_timeseries_deriv!(
#             post_peak_deriv_vect, postpeak_data, reject_spikes)

#         # Copy results into the PulseSummaries object
#         summary.pulse_average[p] = avg
#         summary.pulse_rms[p] = sqrt(abs(s2/Npost - avg*avg))
#         summary.rise_time[p] = rise_time
#         summary.postpeak_deriv[p] = postpeak_deriv
#         summary.peak_index[p] = peak_idx
#         if peak_val > ptm
#             summary.peak_value[p] = peak_val - uint16(ptm)
#         else
#             summary.peak_value[p] = uint16(0)
#         end
#     end
#     summary
# end
using ArrayViews
function mymean(data::Array{Uint16,2}, Nsamp, Npre, timebase)
	summary = PulseSummaries(size(data,2))
	Npost = Nsamp-Npre
    post_peak_deriv_vect = zeros(Float64, Npost)
    timestamp=0
	for p in 1:size(data,2)
        s = s2 = 0.0
		for j = 1:Npre
			d=data[j,p]
			s+=d
			s2+=d*d
		end
        ptm = s/Npre
        summary.pretrig_mean[p] = ptm
        summary.pretrig_rms[p] = sqrt(abs(s2/Npre - ptm*ptm))
        summary.timestamp[p] = timestamp
        s = s2 = 0.0
        peak_idx = 0
        peak_val = uint16(0)
		for j = Npre+1:Nsamp
			d=data[j,p]
            if d > peak_val 
                peak_idx, peak_val = j, d
            end
            d2=d-ptm
			s+=d2
			s2+=d2*d2
		end
        avg = s/Npost
        posttrig_data = data[Npre+2:end,p]
        posttrig_data = view(data, Npre+2:Nsamp,p)
        rise_time::Float64 = estimate_rise_time(posttrig_data, peak_idx-Npre-2,
                                       peak_val, ptm, timebase)

        postpeak_data = data[peak_idx+1:end,p]
        postpeak_data = view(data, peak_idx+1:Nsamp, p)
        const reject_spikes=true
        postpeak_deriv::Float64 = max_timeseries_deriv!(
            post_peak_deriv_vect, postpeak_data, reject_spikes)

        # Copy results into the PulseSummaries object
        summary.pulse_average[p] = avg
        summary.pulse_rms[p] = sqrt(abs(s2/Npost - avg*avg))
        summary.rise_time[p] = rise_time
        summary.postpeak_deriv[p] = postpeak_deriv
        summary.peak_index[p] = peak_idx
        if peak_val > ptm
            summary.peak_value[p] = peak_val - uint16(ptm)
        else
            summary.peak_value[p] = uint16(0)
        end
	end
	summary
end

function fint()
a = 0
for j = 1:10000 a+=j end
a
end
function ffloat()
a = 0.0
for j = 1:10000 a+=j end
a
end
fint();ffloat()
@time fint()
@time ffloat()


        # posttrig_data = sub(data,Npre+2:endof(data))
        # rise_time::Float64 = estimate_rise_time(posttrig_data, peak_idx-Npre-2,
        #                                peak_val, ptm, timebase)

        # postpeak_data = data[peak_idx+1:end]
        # const reject_spikes=true
        # postpeak_deriv::Float64 = max_timeseries_deriv!(
        #     post_peak_deriv_vect, postpeak_data, reject_spikes)

        # # Copy results into the PulseSummaries object
        # summary.pulse_average[p] = avg
        # summary.pulse_rms[p] = sqrt(abs(s2/Npost - avg*avg))
        # summary.rise_time[p] = rise_time
        # summary.postpeak_deriv[p] = postpeak_deriv
        # summary.peak_index[p] = peak_idx
        # if peak_val > ptm
        #     summary.peak_value[p] = peak_val - uint16(ptm)
        # else
        #     summary.peak_value[p] = uint16(0)
        # end

function compute_summary(data, Nsamp, Npre, timebase)
	summary = PulseSummaries(size(data,2))
    Npost = Nsamp-Npre
    timestamp = 0
    post_peak_deriv_vect = zeros(Float64, Npost)
    for p in 1:size(datain,2)
        s = s2 = 0.0
        for j = 1:Npre
            d = data[j,p]
            s += d
            s2 += d*d
        end
        ptm = s/Npre
        summary.pretrig_mean[p] = ptm
        summary.pretrig_rms[p] = sqrt(abs(s2/Npre - ptm*ptm))
        summary.timestamp[p] = timestamp
        s = s2 = 0.0
        peak_idx = 0
        peak_val = uint16(0)
        for j = Npre+1:Nsamp
            d = data[j,p]
            if d > peak_val 
                peak_idx, peak_val = j, d
            end
            d = d-ptm
            s += d
            s2 += d^2
        end
        avg = s/Npost
    end
    summary
end


function estimate_rise_time(pulserecord, peakindex::Integer,peakval,ptm,frametime)
    idx10 = 1
    (peakindex > length(pulserecord) || peakindex < 1) && (peakindex = length(pulserecord))

    idx90 = peakindex
    thresh10 = 0.1*(peakval-ptm)+ptm
    thresh90 = 0.9*(peakval-ptm)+ptm
    for j = 2:peakindex
        pulserecord[j] < thresh10 && (idx10 = j)
        pulserecord[j] > thresh90 && (idx90 = j-1)
    end
    dt = (idx90-idx10)*frametime
    dt * (peakval-ptm) / (pulserecord[idx90]-pulserecord[idx10])
end

# Estimate the derivative (units of arbs / sample) for a pulse record or other timeseries.
# This version uses the default kernel of [-2,-1,0,1,2]/10.0
#
max_timeseries_deriv!(deriv, pulserecord, reject_spikes::Bool) =
    max_timeseries_deriv!(deriv, pulserecord, convert(Vector{eltype(deriv)},[.2 : -.1 : -.2]), reject_spikes)


# Post-peak derivative computed using Savitzky-Golay filter of order 3
# and fitting 1 point before...3 points after.
#
max_timeseries_deriv_SG!(deriv, pulserecord, reject_spikes::Bool) =
    max_timeseries_deriv!(deriv, pulserecord, [-0.11905, .30952, .28572, -.02381, -.45238],
                            reject_spikes)

# Estimate the derivative (units of arbs / sample) for a pulse record or other timeseries.
# Caller pre-allocates the full derivative array, which is available as deriv.
# Returns the maximum value of the derivative.
# The kernel should be a short *convolution* (not correlation) kernel to be convolved
# against the input pulserecord.
# If reject_spikes is true, then the max value at sample i is changed to equal the minimum
# of the values at (i-2, i, i+2). Note that this test only makes sense for kernels of length
# 5 (or less), because only there can it be guaranteed insensitive to unit-length spikes of
# arbitrary amplitude.
#
function max_timeseries_deriv!{T}(
        deriv::Vector{T},       # Modified! Pre-allocate an array of sufficient length
        pulserecord, # The pulse record (presumably starting at the pulse peak)
        kernel::Vector{T},      # The convolution kernel that estimates derivatives
        reject_spikes::Bool  # Whether to employ the spike-rejection test
        )
    N = length(pulserecord)
    Nk = length(kernel)
    @assert length(deriv) >= N+1-Nk
    if Nk > N
        return 0.0
    end
    if Nk+4 > N
        reject_spikes = false
    end

    for i=1:N-Nk+1
        deriv[i] = 0
        for j=1:Nk
            deriv[i] += pulserecord[i+Nk-j]*kernel[j]
        end
    end
    for i=N-Nk+2:length(deriv)
        deriv[i]=deriv[N-Nk+1]
    end
    if reject_spikes
        for i=3:N-Nk-2
            if deriv[i] > deriv[i+2]
                deriv[i] = deriv[i+2]
            end
            if deriv[i] > deriv[i-2]
                deriv[i] = deriv[i-2]
            end
        end
    end
    maximum(deriv)
end




datain = rand(Uint16, (520, 100000))
mymean(datain, 520, 120, 0)
@time mymean(datain, 520, 120, 0)
@time mean(datain);
@time compute_summary(datain, 520, 120, 0)
# @time compute_summary(datain, 520, 120, 0)
@time PulseSummaries(size(datain,2))