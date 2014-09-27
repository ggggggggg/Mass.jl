@everywhere using Mass

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
jld=4
while true
	ljhname, ljhopen = MicrocalFiles.matter_writing_status()
	jldname = hdf5_name_from_ljh(ljhname)
	jld = jldopen(jldname,isfile(jldname) ? "r+" : "w")
	# jld = jldopen(jldname,"w")
	println(names(jld))
	if ljhopen && length(names(jld)) ==0 # has not been initialized
		#initialized h5 file
		println("initializing $jld")
		ljhavailablechannels = MicrocalFiles.ljhallchannels(ljhname)
		println("$(length(ljhavailablechannels)) channels available")
		cg = init_channels(jld, ljhname, ljhavailablechannels[1:min(240, length(ljhavailablechannels))])
		for g in cg
			g["pretrigger_mean_correction"] = [2000.0, 0.1] # imagine this is generated by another step that runs only once
			g["pretrig_rms_lims"] = [0,50]
			g["postpeak_deriv_lims"] = [-1000,20]
		end
		h5step_add(jld, summarize_step)
		h5step_add(jld, ptm_correction_step)
		h5step_add(jld, SelectingStep("pretrig_rms_lims", "pretrig_rms"))
		h5step_add(jld, SelectingStep("postpeak_deriv_lims", "postpeak_deriv"))
		h5step_add(jld, SelectingStepGood(["pretrig_rms", "postpeak_deriv"]))
		h5step_add(jld, Mass.H5Flow.ThresholdStep("selections/good_count", 1000, calibrate_step)) # need to make threshold on number of good pulses
		h5step_add(jld, apply_calibration_step)
	end
	tnow, tlast = time(), tnow
	tnow-tlast < 1 && sleep(1)
	if ljhopen
		println("update!")
		update!(jld,10000)
	else
		println("matter says no ljh is open, last ljh was $ljhname")
	end
	close(jld)
end
