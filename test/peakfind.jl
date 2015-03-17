using KernelDensity, Distributions, PyPlot




function findpeaks(x, fwhm)
	# find indicies in x that correspond to peaks when x is convolved with a gaussian with FWHM=fwhm in units of x's indicies
	sigma = fwhm/2.355
	k=conv(UnivariateKDE(1:length(x),x),Normal(0,sigma))
	localmaximainds(k.density)
end

function localmaximainds(x)
	# finds indicies where x[i] is greater than the 4 nearest neighbors, and a minimum value
	# returns the indicies sorted by the value of x
	minval = 1e-14 # based on observed noise in conv at 1e-17 level with well separated peaks and no background
	lmi = Int[]
	assert(length(x)>5)
	latestlocalminima = realmax(eltype(x))
	@inbounds for i = 3:length(x)-2
		if x[i]>x[i+1] && x[i]>x[i+2] && x[i]>x[i-1] && x[i]>x[i-2] && x[i]>(1.5*latestlocalminima+minval)
			push!(lmi,i)
		elseif x[i]<x[i+1] && x[i]<x[i+2] && x[i]<x[i-1] && x[i]<x[i-2] 
			latestlocalminima = x[i]
		end
	end
	return lmi[sortperm(x[lmi])]
end

# generate a histogram with known peak locations
d = float(rand(Poisson(10),30000))
x = 1:1:length(d)
peaklocs_true = rand(x,10)
d[peaklocs_true] = 100

fwhm = 10
k = UnivariateKDE(x,d)
k2 = conv(k,Normal(0,fwhm/2.355)) # convolve with a gaussia with mean 0, sigma=10 in units of k.x

peaklocs_found = findpeaks(d,fwhm)
println(Set(peaklocs_found[1:length(peaklocs_true)])==Set(peaklocs_true))

plot(x,k2.density)
plot(peaklocs_found, k2.density[peaklocs_found],"o")
