module LJHTest
using Mass, Mass.MicrocalFiles,Base.Test
fnames = ["/Volumes/Drobo/exafs_data/20140719_ferrioxalate_pump_probe/20140719_ferrioxalate_pump_probe_chan$d.ljh" for d in 1:2:5]
ljhgroup = microcal_open(fnames)
f1,f2,f3 = ljhgroup.ljhfiles


@time [p for p in f1[1:end]]
@time Mass.MicrocalFiles.LJH.fileData(f1)

gs = ljhgroup[1:100]
@time [p for p in gs]

a=[p for p in f1[1:100]]
a2 = [p for p in ljhgroup[1:100]]
@test a == a2
a3 = [p for p in ljhgroup[1:end]]
for j=1:10
@test f2[j] == ljhgroup[j+length(f1)]
end
@test [p for p in ljhgroup[length(f1)+1:length(f1)+10]] == [p for p in f2[1:10]]
@test [p for p in ljhgroup[length(f1)+1:length(f1)+10]] != [p for p in f3[1:10]]
@test [p for p in ljhgroup[length(f1)+length(f2)+1:length(f1)+length(f2)+10]] == [p for p in f3[1:10]]

singleljhgroup = Mass.MicrocalFiles.LJH.LJHGroup(f1.name)
singleljhgroup2 = Mass.MicrocalFiles.LJH.LJHGroup(f1)
@test singleljhgroup[7] == singleljhgroup2[7]
@test length(singleljhgroup) == length(singleljhgroup2)
@test [p for p in singleljhgroup[55:734]] == [p for p in f1[55:734]]
@test filenames(ljhgroup) == fnames
record_nsamples(ljhgroup)
pretrig_nsamples(ljhgroup)
@test_throws ErrorException channel(ljhgroup)
frametime(ljhgroup)


end