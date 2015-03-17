using Mass, ArrayViews, Dierckx

@everywhere using PyCall

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

function apply(cal::Calibration, pulse_rms)
	energy = pulse_rms.*(cal.energies[end]/cal.estimates[end])
	return energy
end
apply_calibration_step = Step(apply, "calibration/pulse_rms", "pulse_rms", [], "energy")

select_good(pt_rms, postpeak_deriv, peak_index) = (0.0 .< pt_rms .< 25.0)  & (0.0 .< postpeak_deriv .< 40.0)  & (410 .< peak_index .< 425)
select_step = SelectingStep(Step(select_good,[],["pretrig_rms", "postpeak_deriv", "peak_index"],[],["selections/good"]))

function calc_average_pulse_filter(ljhname, good, pulse_height)
	use = (abs(pulse_height/median(pulse_height)-1) .< 0.1) & good
	ljh = microcal_open(ljhname)
	inds = find(use)

	average_pulse = zeros(Float64, record_nsamples(ljh))
	for i = 1:length(use)
		average_pulse+=ljh[i][1]
	end
	close(ljh)
	average_pulse /= length(use)
	average_pulse -= mean(average_pulse)
	average_pulse /= dot(average_pulse, average_pulse)
end
avg_pulse_step = Mass.H5Flow.ThresholdStep("selections/good_count", 5000, Step(calc_average_pulse_filter,["pulsefile_names"], ["selections/good", "pulse_rms"], "average_pulse_filter", []))

function filter1lag(r::UnitRange, ljhname, filter, npulses)
	ljh = microcal_open(ljhname)
	out = Float64[dot(ljh[i][1], filter) for i in r]
	close(ljh)
	out
end
filter_step = Mass.H5Flow.RangeStep(filter1lag, ["pulsefile_names", "average_pulse_filter"], "npulses", [], ["filt_value"]) #npulses isn't used, it just acts as an input size

function addhist(prevbinedges, prevcounts, newenergies)
	binedges, addcounts = hist(newenergies, prevbinedges)
	newcounts = prevcounts + addcounts
	newcounts	
end
hist_step = SelectedStep(addhist, "good",["filt_value_bin_edges", "filt_value_bin_counts"], "filt_value", "filt_value_bin_counts", ())
hist_step_bin_edges = [realmin(Float64),[0:0.001:2], realmax(Float64)]
hist_step_bin_counts = zeros(Int64, length(hist_step_bin_edges)-1)


### CALIBRATION
function peakassign(locations_arb, locations_true)
# locations_arb is an array of possible peak locations in arbitrary units, assumed to have an unknown, 
# nonlinear (but monotonic and smooth) relationship to locations_true
# locations_arb may contain more peaks than locations_true, including some spurious peaks
# the goal is to find the best possible set of assignments for entries in locations_arb to 
# correspond to locations_true by examining all possible assignments and calcualting some figure of merit
@assert(issorted(locations_arb))
@assert(issorted(locations_true))
@assert(length(locations_arb)>=length(locations_true))
combos = combinations(locations_arb, length(locations_true))
length(combos)==1 && return first(combos)

best_combo = first(combos)
best_fom = realmax(Float64)
for c in combos
	fom = peakassign_fig_of_merit(c, locations_true)
	if fom < best_fom
		best_fom = fom
		best_combo = c
	end
end
best_combo, best_fom
end

function peakassign_fig_of_merit(x, y)
	# average fractional error trying to predict the y value of the middle of 3 points using the slope
	# from the outer two points and the x value of the middle point
	err = 0.0
	for i = 1:length(y)-2
		yip1_prediction = y[i]+(y[i+2]-y[i])*(x[i+1]-x[i])/(x[i+2]-x[i])
		err += abs(1-y[i+1]/yip1_prediction)
	end
	err /= length(y-2)
end

@pyimport scipy.signal as scipysignal
function findpeaks(y, length_scales; min_snr = 3, min_dist=-1)
	# find peaks using scipy.signal.find_peaks_cwt
	# then make sure any reported peaks are local maxima
	# returnds indicies of peaks orderes from lowest to highest y value
	raw_peakinds = int(scipysignal.find_peaks_cwt(y, length_scales, min_snr=min_snr))
	peakinds = Int[]
	min_dist == -1 && (min_dist = int(maximum(length_scales)/2))
	for p in raw_peakinds
		r = max(1,p-min_dist):min(length(y), p+min_dist)
		local_max_ind = indmax(y[r])+first(r)-2
		local_max_ind in peakinds || push!(peakinds, local_max_ind)
	end
	perm = sortperm(y[peakinds])
	peakinds[perm]
end
function findpeaks_wrapped(y,bin_edges, not_used) 
	peak_inds = findpeaks(y,[2,4,8,16,32])
	bin_centers = bin_edges[2]:bin_edges[3]:bin_edges[end-2]+0.5*bin_edges[3]
	peak_x = bin_centers[peak_inds]
	peak_inds, peak_x
end
findpeaks_step = Mass.H5Flow.ThresholdStep("selections/good_count", 5000, Step(findpeaks_wrapped,["filt_value_bin_counts","filt_value_bin_edges"],["selections/good"], ["filt_value_bin_counts_peak_inds","filt_value_peaks_locations"], []))

function peakassign_wrapped(locations_arb, locations_true)
	use_arb_locations = locations_arb[max(end-length(locations_true)-4,1):end]
	arbs = sort!(use_arb_locations)
	assigns, fom = peakassign(arbs, locations_true)
end
assignpeaks_step = Step(peakassign_wrapped,["filt_value_peaks_locations","calibration_energies"],[], ["filt_value_calibration_energy_matches","filt_value_calibration_energy_matches_fom"], [])

function calc_spline(x,y)
	spl = Spline1D(float(x),float(y),k=1,bc="extrapolate")
end
calc_spline_step = Step(calc_spline, ["filt_value_calibration_energy_matches","calibration_energies"],[],["calibration_spline"],[])

function apply_spline(spl, energy_estimator)
	energy = evaluate(spl, energy_estimator)
end
apply_spline_step = Step(apply_spline, ["calibration_spline"],["filt_value"],[],["energy"])


hist_step_energy = SelectedStep(addhist, "good",["energy_value_bin_edges", "energy_value_bin_counts"], "energy", "energy_value_bin_counts", ())
hist_step_energy_bin_edges = [realmin(Float64),[0:2:20000], realmax(Float64)]
hist_step_energy_bin_counts = zeros(Int64, length(hist_step_energy_bin_edges)-1)


end # everywhere

tnow, tlast = time(), time()
tstart = time()
println("starting loop!")
pulse_steps_done = 1 # initializat with any nonzero value
max_chans = 2
ljhname, ljhopen = "/Volumes/Drobo/exafs_data/20141030_brown_ferrioxalate_1M_calibronium/20141030_brown_ferrioxalate_1M_calibronium_chan1.ljh", false
close(jldopen(hdf5_name_from_ljh(ljhname),"w")) #wipe the file
j=0
while pulse_steps_done > 0 && j<4
	j+=1
	#ljhname, ljhopen = MicrocalFiles.matter_writing_status()
	ljhname, ljhopen = "/Volumes/Drobo/exafs_data/20141030_brown_ferrioxalate_1M_calibronium", false
	jldname = hdf5_name_from_ljh(ljhname)
	# jldname = "/Volumes/Drobo/exafs_data/20141008_brown_ferrioxalate_straw_2mm_emission/20141008_brown_ferrioxalate_straw_2mm_emission_mass_julia.hdf5"
	jldopen(jldname,isfile(jldname) ? "r+" : "w") do jld
		chans_to_init = 1:2:2*max_chans
		cg = init_channels(jld, ljhname, chans_to_init)
		length(cg) > 0 && println("initialized $(length(cg)) channels in file $jldname from channels $chans_to_init")
		for c in cg
			h5step_add(c, summarize_step)
			h5step_add(c, select_step)
			h5step_add(c, avg_pulse_step)
			h5step_add(c, filter_step)
			h5step_add(c, hist_step)
			h5step_add(c, findpeaks_step)
			h5step_add(c, assignpeaks_step)
			h5step_add(c, calc_spline_step)
			h5step_add(c, apply_spline_step)
			h5step_add(c, hist_step_energy)
			c["filt_value_bin_edges"] = hist_step_bin_edges
			c["filt_value_bin_counts"] = hist_step_bin_counts
			c["calibration_energies"] = [4090,4952,5898,6403,6929,8040]
			c["energy_value_bin_edges"] = hist_step_energy_bin_edges
			c["energy_value_bin_counts"] = hist_step_energy_bin_counts
		end
	end
	tnow, tlast = time(), tnow
	sleeptime = 1-(tnow-tlast)
	sleeptime > 0 && sleep(sleeptime)
	jld = jldopen(jldname,"r+")
	pulse_steps_done = update!(jld, 10000)
	close(jld)
	println("outer loop")
end
println("finished loop because pulse_steps_done = $pulse_steps_done")

ljhname, ljhopen = "/Volumes/Drobo/exafs_data/20141030_brown_ferrioxalate_1M_calibronium/20141030_brown_ferrioxalate_1M_calibronium_chan1.ljh", false
jld = jldopen(hdf5_name_from_ljh(ljhname),"r+")
c = jld["chan1"]
ljh = microcal_open(ljhname)
good = read(c["selections/good"]).==1
ppd = read(c["postpeak_deriv"])
ptrms= read(c["postpeak_deriv"])

a=[ljh[i][1] for i in find(good[1:1000])]

using PyPlot
energy = c["energy"][:]
plot(energy[good],".")
xlabel("pulse number (arb)")
ylabel("energy of good pulses")


# plot spectra to check calibration
figure()
title("energy 2 channels")
edges = c["energy_value_bin_edges"][:]
counts = c["energy_value_bin_counts"][:]
bin_centers = edges[2]:edges[3]:edges[end-2]+0.5*edges[3]
plot(bin_centers, counts[2:end-1])
xlabel("energy")
ylabel("counts/bin")
c2 = jld["chan3"]
plot(bin_centers, c2["energy_value_bin_counts"][2:end-1])
edges_fv = c["filt_value_bin_edges"][:]
bin_centers_fv = edges_fv[2]:edges_fv[3]:edges_fv[end-2]+0.5*edges_fv[3]


# plot for checking peak finding and assingment
figure()
for cc in (jld["chan1"],jld["chan3"])
	counts = cc["filt_value_bin_counts"][:]
	peakinds = findpeaks(counts, [2,4,8,16,32], min_snr=4)

	energies = vcat(cc["calibration_energies"][:])
	usepeakinds = peakinds[end-length(energies)-4:end]
	arbs = sort!(bin_centers_fv[usepeakinds])
	assigns, fom = peakassign(arbs, energies)
	assigninds = indexin(assigns, bin_centers_fv)

	plot(bin_centers_fv, counts[2:end-1])
	plot(bin_centers_fv[peakinds], counts[2:end-1][peakinds],"o")
	plot(bin_centers_fv[usepeakinds], counts[2:end-1][usepeakinds],"s")
	plot(bin_centers_fv[assigninds], counts[2:end-1][assigninds],"v")
end

# test spline

# spl = read(c["calibration_spline"])
# counts, edges = plt.hist(evaluate(spl, c["filt_value"][:][good]),0:4:20000,histtype="step")
# plt.hist(c["filt_value"][:][good]*6937,0::20000,histtype="step")
# x=4
# try
# figure()
# plot(c["filt_value_calibration_energy_matches"][:],c["calibration_energies"][:],".")
# plot(bin_centers[assigninds],c["calibration_energies"][:],"s")
# spl = Spline1D(c["filt_value_calibration_energy_matches"][:],float(c["calibration_energies"][:]),k=1,bc="extrapolate")
# plot(c["filt_value_bin_edges"][:],evaluate(spl, c["filt_value_bin_edges"][:]))
# catch
# end	