using Mass, Mass.H5Flow, Base.Test

h = jldopen("test.h5","w")
g = g_require(h, "g1")


@test a_read(g,"attribute",0) == 0
a_require(g,"attribute", 55)
@test a_read(g, "attribute", 0)==55
@test a_read(g, "attribute") == 55
@test a_require(g, "attribute", 55) == 55
@test_throws ErrorException a_require(g, "attribute", 4) == 55
a_update(g, "attribute", "string")
@test a_read(g, "attribute") == "string"
a_update(g, "new_attribute", 77)
@test a_read(g, "new_attribute")==77
for j = 1:2
a_update(g, "new_attribute", j)
@test a_read(g, "new_attribute",0)==j
end	
function f()
	j = a_read(g,"new_attribute",0)
	a_update(g, "new_attribute", j+1)
	@test j+1 == a_read(g,"new_attribute",0)
end
for j=1:3 f() end

d_update(g, "dataset", [1:10])
d_extend(g, "dataset", [11:20], 11:20)
@test g["dataset"][:] == [1:20]
@test length(g["dataset"]) == 20
d_extend(g, "dataset", [21:30])
@test g["dataset"][:] == [1:30]
@test length(g["dataset"]) == 30
@test length(g["dataset"]) == 30
d_update(g,"dataset",[1:10])
@test g["dataset"][:] == [1:10]

### selections ###
selections = ["good", "pretrig_mean", "postpeak_deriv"]
[H5Flow.selection_extend(g, joinpath("selections",n), [true],1:1) for n in selections]
a = [randbool() for j=1:30]
assert([1, 1, 1] == H5Flow.selection_lengths(g))
H5Flow.selection_extend(g, "good", a,1:30)
assert(a ==H5Flow.selection_read(g, "good", 1:30))
H5Flow.selection_extend(g, "good", [true for j=1:10])