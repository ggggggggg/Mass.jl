using Mass
fname = "/Users/oneilg/Desktop/livetestljh/20140820_104348_chan1.ljh"
ljhgroup=microcal_open(fname)

h5 = jldopen(hdf5_name_from_ljh(ljhgroup),"w")
close(h5)
h5 = jldopen(hdf5_name_from_ljh(ljhgroup),"r+")

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
function calibrate(pulse_rms)
	println("Calibration!!!!**!!")
	@show (length(pulse_rms))
	(Calibration(["Zero","median"], [0, 8000], [0,median(pulse_rms)]),)
end
calibrate_step = SelectedStep(calibrate, "good", [], "pulse_rms", "calibration/pulse_rms")

function apply(cal::Calibration, pulse_rms)
	energy = pulse_rms.*(cal.energies[end]/cal.estimates[end])
	return energy
end
apply_calibration_step = Step(apply, "calibration/pulse_rms", "pulse_rms", [], "energy")

g = init_channel(h5, ljhgroup)
g["pretrigger_mean_correction"] = [2000.0, 0.1] # imagine this is generated by another step that runs only once
g["pretrig_rms_lims"] = [0,50]
g["postpeak_deriv_lims"] = [-1000,20]
h5step_add(g, summarize_step)
h5step_add(g, ptm_correction_step)
h5step_add(g, SelectingStep("pretrig_rms_lims", "pretrig_rms"))
h5step_add(g, SelectingStep("postpeak_deriv_lims", "postpeak_deriv"))
h5step_add(g, SelectingStepGood(["pretrig_rms", "postpeak_deriv"]))
h5step_add(g, Mass.H5Flow.ThresholdStep("pulse_rms", 1000, calibrate_step))
h5step_add(g, apply_calibration_step)
for j=1:5 update!(g,10000) end

getgood(g,s="good") = reinterpret(Bool, g["selections/$s"][:])
using PyPlot
function histenergy(g)
	energy = g["energy"][:][getgood(g)]
	plt.hist(energy, [0:5:10000])
	xlabel("energy (eV)")
	ylabel("counts per 5 eV bin")
	title(name(g))
end
histenergy(g)
figure()
plot(g["pretrig_rms"][:][getgood(g, "good")],".")
figure()
plot(g["timestamp"][:],".")

# pythonattrs = ["npulses", "mass_version", "timebase", "channel", "git_state", "julia_version","pulsefiles_names","pulsefile_lengths"]
# Mass.H5Flow.pythonize(g,pythonattrs,pythonattrs)