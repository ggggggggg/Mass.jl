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
avg_pulse_step = Mass.H5Flow.ThresholdStep("selections/good_count", 1000, Step(calc_average_pulse_filter,["pulsefile_names"], ["selections/good", "pulse_rms"], "average_pulse_filter", []))

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
			c["filt_value_bin_edges"] = hist_step_bin_edges
			c["filt_value_bin_counts"] = hist_step_bin_counts
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
pfilt = c["filt_value"][:]
plot(pfilt[good],".")
xlabel("pulse number (arb)")
ylabel("filt_value (median=1)")

edges = c["filt_value_bin_edges"][:]
counts = c["filt_value_bin_counts"][:]
bin_centers = edges[2]:edges[3]:edges[end-2]+0.5*edges[3]
figure()
plot(bin_centers, counts[2:end-1])
xlabel("filt_value")
ylabel("counts/bin")

