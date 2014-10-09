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
pulse_steps_done = 1 # initializat with any nonzero value
max_chans = 240
ljhname = ""

try 
	while true
		old_ljhname = ljhname
		ljhname, ljhopen = MicrocalFiles.matter_writing_status()
		if old_ljhname != ljhname
			try 
				close(jld) 
				print("closing old jld file $jld")
			end
			jldname = hdf5_name_from_ljh(ljhname)
			jld = jldopen(jldname, "w")
			println("opening new jld file $jld")

	    end
		chans_to_init = 1:2:2*max_chans
		cg = init_channels(jld, ljhname, chans_to_init)
		length(cg)>0 && println("initilizeing $(length(cg)) channels in $jld")
		for chan in cg h5step_add(chan, summarize_step) end

		tnow, tlast = time(), tnow
		sleeptime = 0.1-(tnow-tlast)
		sleeptime > 0 && sleep(sleeptime)
		pulse_steps_done = update!(jld,10000) # does the majority of the processing

	end

finally
	close(jld)
end