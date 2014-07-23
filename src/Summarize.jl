# Functions to perform a summarize data step on a single channel's LJH file.
# Joe Fowler, NIST
# July 18, 2014

module Summarize
export summarize, PulseSummaries
using ..MicrocalFiles

using HDF5, ..H5Helper

# Contain a single channel's complete "pulse summary information"
# We use these summary data for:
# 1) Making cuts against pathological data records
# 2) Having quick indicators of pulse energy and arrival time
# 3) Making corrections for systematic errors (e.g., the correlation between gain and
#    pretrigger mean value).
#
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






# Generate the HDF5 summary for an LJH file by filename
summarize(filename::String) = summarize(MicrocalFiles.LJHFile(filename))



# Generate the HDF5 summary for an LJH file given an open LJHFile objects
function summarize(file::MicrocalFiles.LJHFile)
    hdf5name = hdf5_name_from_ljh_name(file.name)
    println("We are about to summarize file into '$hdf5name'")
    if isreadable(hdf5name)
        h5file = h5open(hdf5name, "r+")
    else
        h5file = h5open(hdf5name, "w")
    end
    grpname=string("chan$(file.channum)")
    h5grp = g_create_or_open(h5file, grpname)

    # Store basic information
    a_update(h5grp, "npulses", file.nrec)
    a_update(h5grp, "nsamples", file.nsamp)
    a_update(h5grp, "npresamples", file.npre)
    a_update(h5grp, "frametime", file.dt)
    a_update(h5grp, "rawname", file.name)

    summary = compute_summary(file)

    summgrp = g_create_or_open(h5grp,"summary")
    for field in names(summary)
        d_update(summgrp, string(field), getfield(summary, field))
#         summgrp[string(field)] = getfield(summary, field)
        println(string("Updating HDF5 with $grpname/summary/", field))
    end
    close(h5file)
end



compute_summary(filename::String) = compute_summary(microcal_open(filename))

# Compute the per-pulse data summary. This function returns a PulseSummaries
# object given an open LJHFile object. It does not know anything about HDF5
# files.
function compute_summary(file::MicrocalFiles.LJHFile)
    summary = PulseSummaries(file.nrec)
    Npre, Npost = file.npre+2, file.nsamp-(file.npre+2)
    post_peak_deriv_vect = zeros(Float64, Npost)
    for (p, (data, timestamp)) in enumerate(file)
        # Pretrigger computation first
        s = s2 = 0.0
        for j = 1:Npre
            d = data[j]
            s += d
            s2 += d*d
        end
        ptm = s/Npre
        summary.pretrig_mean[p] = ptm
        summary.pretrig_rms[p] = sqrt(s2/Npre - ptm*ptm)

        # Now post-trigger calculations
        s = s2 = 0.0
        peak_idx = 0
        peak_val = uint16(0)
        for j = Npre+1:file.nsamp
            d = data[j]
            if d > peak_val 
                peak_idx, peak_val = j, d
            end
            d = d-ptm
            s += d
            s2 += d^2
        end
        avg = s/Npost

        posttrig_data = sub(data,Npre+2:endof(data))
        rise_time = estimate_rise_time(posttrig_data, peak_idx-Npre-2,
                                       peak_val, ptm, file.dt)

        postpeak_data = data[peak_idx+1:end]
        const reject_spikes=true
        postpeak_deriv = max_timeseries_deriv!(
            post_peak_deriv_vect, postpeak_data, reject_spikes)

        # Copy results into the PulseSummaries object
        summary.pulse_average[p] = avg
        summary.pulse_rms[p] = sqrt(s2/Npost - avg*avg)
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


# Rise time computation
# We define rise time based on rescaling the pulse so that pretrigger mean = 0
# and peak value = 100%. Then use the linear interpolation between the first
# point exceeding 10% and the last point not exceeding 90%. The time it takes that
# interpolation to rise from 0 to 100% is the rise time.
#
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
max_timeseries_deriv!(deriv, pulserecord::Array, reject_spikes::Bool) =
    max_timeseries_deriv!(deriv, pulserecord, convert(Vector{eltype(deriv)},[.2 : -.1 : -.2]), reject_spikes)


# Post-peak derivative computed using Savitzky-Golay filter of order 3
# and fitting 1 point before...3 points after.
#
max_timeseries_deriv_SG!(deriv, pulserecord::Vector, reject_spikes::Bool) =
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
        pulserecord::Vector, # The pulse record (presumably starting at the pulse peak)
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



# Given an LJH file name, return the HDF5 name
# Generally, /x/y/z/data_taken_chan1.ljh becomes /x/y/z/data_taken_mass.hdf5
function hdf5_name_from_ljh_name(ljhname::String)
    dir = dirname(ljhname)
    base = basename(ljhname)
    path,suffix = splitext(ljhname)
    m = match(r"_chan\d+", path)
    path = string(path[1:m.offset-1], "_mass.hdf5")
end
end # endmodule