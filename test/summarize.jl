using Mass
fname1 = "/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan1.ljh"
fname2 = "/Volumes/Drobo/exafs_data/20140720_ferrioxalate_pump_probe/20140720_ferrioxalate_pump_probe_chan1.ljh"
ljhgroup1=microcal_open([fname1,fname2])
ljhgroup1.ljhfiles[end].nrec = div(ljhgroup1.ljhfiles[end].nrec,2)
ljhgroup2 = Mass.MicrocalFiles.LJHGroup(ljhgroup1.ljhfiles)

h5 = jldopen(hdf5_name_from_ljh(ljhgroup2),"w")
close(h5)
h5 = jldopen(hdf5_name_from_ljh(ljhgroup2),"r+")

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
	(Calibration(["Zero","MnKAlpha"], [0, 5898], [0,median(pulse_rms)]),)
end
calibrate_step = SelectedStep(calibrate, "good", [], "pulse_rms", "calibration/pulse_rms")

function apply(cal::Calibration, pulse_rms)
	energy = pulse_rms.*(cal.energies[end]/cal.estimates[end])
	return energy
end
apply_calibration_step = Step(apply, "calibration/pulse_rms", "pulse_rms", [], "energy")

g = init_channel(h5, ljhgroup2)
g["pretrigger_mean_correction"] = [2000.0, 0.1] # imagine this is generated by another step that runs only once
g["pretrig_rms_lims"] = [0,50]
g["postpeak_deriv_lims"] = [-1000,20]
h5step_add(g, summarize_step)
h5step_add(g, ptm_correction_step)
h5step_add(g, SelectingStep("pretrig_rms_lims", "pretrig_rms"))
h5step_add(g, SelectingStep("postpeak_deriv_lims", "postpeak_deriv"))
h5step_add(g, SelectingStepGood(["pretrig_rms", "postpeak_deriv"]))
h5step_add(g, Mass.H5Flow.ThresholdStep("pulse_rms", 400000, calibrate_step))
h5step_add(g, apply_calibration_step)
for j=1:5 update!(g,300000) end

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