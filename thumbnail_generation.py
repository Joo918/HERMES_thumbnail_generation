from paraview import lookuptable
import os
import numpy as np
import random
from string import Template
import shutil

###########User Input Fields##########

#particle type
particle_type = "K+"
#data file name
data_file_name = "K+Filtered_HERMES+C.csv"
#name of the export csv. Particle type is appended as a prefix
export_file_name = "thumbnail"
#import file location (location of data files)
data_location = "C:/Users/ajido/Desktop/vt stuff/info visualization/project"
#export file location
export_location = "C:/Users/ajido/Desktop/vt stuff/info visualization/project/HERMES_thumbnail_generation/Export"
#column names to generate thumbnails by
interest_columns = [
	"x",
	"y",
	"P_hperp"
]
#data to keep for glee's parametric interaction
cols_to_preserve = [
	"Q^2",
	"x",
	"y",
	"Phi_s",
	"epsilon",
	"Z(K+/-)",
	"P_hperp",
	"Phi_h"
]
#qzp value to filter by
qzp_filter_val = 0.3
#boolean. If true, filter qzp > qzp_filter_val. If false, <=
filter_above = 0
#number of thumbnails to generate
thumbnail_num = 6

x_axis_labels = [0, 0.5, 1, 1.5]
y_axis_labels = [0, 0.5, 1, 1.5, 2]
scale_arr = [1.5, 1, 1]
#####################################

#remove any tmp files
tmpfname = export_location + "/tmp.txt"
if os.path.exists(tmpfname):
	os.remove(tmpfname)

exportDirPath = export_location + "/" + particle_type + export_file_name
if os.path.exists(exportDirPath):
	shutil.rmtree(exportDirPath)
os.mkdir(exportDirPath)
	
view = GetActiveView()
#create a background image
backbox = Box()
backbox.XLength = 2
backbox.YLength = 3.2
backbox.ZLength = 1
backbox.Center = [0.75, 1.0, -1.0]
dpbox = GetDisplayProperties()
dpbox.DiffuseColor = [0.0, 0.0196078431372549, 0.403921568627451]
Show(backbox)
Render()

view.CameraPosition[2] *= 0.5
view.CameraPosition[1] -= 0.2
view.OrientationAxesVisibility = 0
	
#get data
reader = CSVReader(FileName= data_location + '/' + data_file_name)
Hide()
#filter by qzp
#sort by first variable
qzp_filter = Template('''

import os
import numpy as np
import random

print("filtering by qzp!")
qzp = self.GetInput().GetColumnByName('(q/zp)^2')
	
sort_col_name = "$scn"
sort_col_index = 0

newlist = []
for i in range(27):
	newlist.append([])
	if self.GetInput().GetColumnName(i) == sort_col_name:
		sort_col_index = i

filter_val = $fv
eval_over = $vo
samples = random.sample(range(qzp.GetSize()), qzp.GetSize())
for i in range(qzp.GetSize()):
	row_at_i = self.GetInput().GetRow(samples[i])

	if eval_over and qzp.GetValue(samples[i]) > filter_val:
		for j in range(27):
			newlist[j].append(row_at_i.GetValue(j).ToFloat())
	elif (not eval_over) and qzp.GetValue(samples[i]) <= filter_val:
		for j in range(27):
			newlist[j].append(row_at_i.GetValue(j).ToFloat())

final_sorted_data = np.array(newlist, dtype=float)
#sort_indices = np.argsort(final_sorted_data[sort_col_index])
#final_sorted_data = final_sorted_data[:, sort_indices]

for i in range(27):
	output.RowData.append(final_sorted_data[i], self.GetInput().GetColumnName(i))

''')

ppf_init = ProgrammableFilter(reader)
ppf_init.Script = qzp_filter.substitute(scn=interest_columns[0], fv=qzp_filter_val, vo=filter_above)

#set up view and heat color map
view.ViewSize = [350, 400]

lr = lookuptable.vtkPVLUTReader()
lr.Read(data_location + '/HeatColormap.xml')

#generate n thumbnails + export csv

divn_filter = Template('''
import os
import numpy as np
import math
import random

scriptDirectory = "$s_dir"
filenamePrefix = "$f_prefix"
ptype = "$p_type"

filename = scriptDirectory + '/tmp.txt'
dirname = scriptDirectory + '/' + ptype + filenamePrefix
filename3 = scriptDirectory + '/' + ptype + filenamePrefix + '/' + ptype + filenamePrefix + 'Thumbnails.csv'
targetColumns = $t_cols
xycols = $xy_cols


if not os.path.exists(filename):
	f = open(filename, 'w+')
	f.write('0')
	f.close()
	
if not os.path.exists(filename3):
	f = open(filename3, 'w+')
	f.write('Cell')
	for i in range(27):
		#print(self.GetInput().GetColumnName(i))
		if self.GetInput().GetColumnName(i) in targetColumns:
			f.write(',' + self.GetInput().GetColumnName(i))
			f.write(',SD_' + self.GetInput().GetColumnName(i))
	f.write(',image\\n')
	f.close()

n = $n
length = self.GetInput().GetColumnByName(targetColumns[0]).GetSize()
chunk = length / n

fr = open(filename, 'r')
lastN = int(fr.read())
fr.close()

readnumline = open(filename3, 'r')
lines = readnumline.read()
readnumline.close()
arrr = lines.split('\\n')
numl = len(arrr)
imgnum = numl - 2
print('run #' + str(imgnum))

lastrun = False
if length - chunk - lastN < chunk:
	lastrun = True

print('chunk = ' + str(chunk))

x = self.GetInput().GetColumnByName(xycols[0])
y = self.GetInput().GetColumnByName(xycols[1])
newx = []
newy = []
newAttributes = [[] for i in range(len(targetColumns))]

meanvals = [0] * 27
savedvals = [[] for i in range(27)]
stds = [0] * 27

startAt = (imgnum % n) * chunk
for i in range(startAt, startAt + chunk):
	newx.append(x.GetValue(i))
	newy.append(y.GetValue(i))
	ithrow = self.GetInput().GetRow(i)
	for j in range(27):
		if self.GetInput().GetColumnName(j) in targetColumns:
			meanvals[j] += ithrow.GetValue(j).ToFloat()
			savedvals[j].append(ithrow.GetValue(j).ToFloat())
			targetIndex = targetColumns.index(self.GetInput().GetColumnName(j))
			newAttributes[targetIndex].append(ithrow.GetValue(j).ToFloat())
	
for i in range(27):
	meanvals[i] = meanvals[i] / chunk

for i in range(0, chunk):
	for j in range(27):
		if self.GetInput().GetColumnName(j) in targetColumns:
			stds[j] += (meanvals[j] - savedvals[j][i]) ** 2

for i in range(27):
	stds[i] = math.sqrt(stds[i] / chunk)
	
output.RowData.append(np.array(newx, float), 'x_coord')
output.RowData.append(np.array(newy, float), 'y_coord')
for i in range(len(newAttributes)):
	output.RowData.append(np.array(newAttributes[i], float), targetColumns[i] )
#output.RowData.append(np.array(newd, float), 'density')

dat = open(filename3, 'a')
dat.write(str(imgnum))
for i in range(27):
	if self.GetInput().GetColumnName(i) in targetColumns:
		dat.write(',' + str(meanvals[i]))
		dat.write(',' + str(stds[i]))
dat.write(',' + ptype + filenamePrefix + str(imgnum) + '.png\\n')
dat.close()

fw = open(filename, 'w')
val = str(lastN + chunk)
fw.write(val)
fw.close()

if lastrun:
	os.remove(filename)
''')

density_filter = '''
import numpy as np

n = 100
x = self.GetInput().GetColumnByName('x_coord')
y = self.GetInput().GetColumnByName('y_coord')

xrange = x.GetRange()
xmin = xrange[0]
xmax = xrange[1]
xrnge = xmax - xmin

yrange = y.GetRange()
ymin = yrange[0]
ymax = yrange[1]
yrnge = ymax - ymin

cell_dict = {}

newlist = [None] * x.GetSize()
newx = [None] * x.GetSize()
newy = [None] * x.GetSize()

for i in range(x.GetSize()):
	row_at_i = self.GetInput().GetRow(i)
	curx = int((x.GetValue(i) - xmin) * n / xrnge)
	cury = int((y.GetValue(i) - ymin) * n / yrnge)
	newx[i] = x.GetValue(i)
	newy[i] = y.GetValue(i)
	curcoord = (curx, cury)
	
	if cell_dict.get(curcoord) is None:
		cell_dict[curcoord] = [i]
	else:
		cell_dict[curcoord].append(i)

for coord in cell_dict.keys():
	val = len(cell_dict[coord])
	for j in cell_dict[coord]:
		newlist[j] = val

output.RowData.append(np.array(newx, float), 'x')
output.RowData.append(np.array(newy, float), 'y')
output.RowData.append(np.array(newlist, float), 'density')
'''

for i in range(thumbnail_num):
	ppfn = ProgrammableFilter(ppf_init)
	ppfn.Script = divn_filter.substitute(s_dir=export_location, f_prefix=export_file_name, p_type=particle_type, t_cols=cols_to_preserve, n=thumbnail_num, xy_cols=interest_columns[1:])
	
	ppfdn = ProgrammableFilter(ppfn)
	ppfdn.Script = density_filter
	
	ttpn = TableToPoints(ppfdn)
	ttpn.KeepAllDataArrays = True
	ttpn.XColumn = 'x'
	ttpn.YColumn = 'y'
	ttpn.a2DPoints = True
	
	ttpn.UpdatePipeline()
	
	pvin = PointVolumeInterpolator(Input=ttpn, Source='Bounded Volume')
	pvin.UpdatePipeline()
	
	cntrn = Contour(pvin)
	cntrn.ComputeScalars = True
	cntrn.Isosurfaces = [1, 1.2, 1.4, 1.6, 2]
	
	rep = Show(cntrn)
	dp = GetDisplayProperties()
	dp.Representation = 'Wireframe'
	dp.LineWidth = 5
	dp.DataAxesGrid.GridAxesVisibility = 1
	dp.DataAxesGrid.XTitle = interest_columns[1]
	dp.DataAxesGrid.YTitle = interest_columns[2]
	
	dp.DataAxesGrid.XAxisLabels = x_axis_labels
	dp.DataAxesGrid.XAxisUseCustomLabels = 1
	dp.DataAxesGrid.YAxisLabels = y_axis_labels
	dp.DataAxesGrid.YAxisUseCustomLabels = 1
	
	dp.DataAxesGrid.GridColor = [0.5019607843137255, 0.9529411764705882, 1.0]
	dp.DataAxesGrid.AxesToLabel = 7
	dp.DataAxesGrid.ShowGrid = 1
	dp.DataAxesGrid.ShowEdges = 0
	dp.DataAxesGrid.ShowTicks = 0
	
	dp.DataAxesGrid.XTitleBold = 1
	dp.DataAxesGrid.YTitleBold = 1
	dp.DataAxesGrid.XTitleFontSize = 16
	dp.DataAxesGrid.YTitleFontSize = 16
	
	dp.Scale = scale_arr
	
	rep.ColorArrayName = 'density'
	arr = cntrn.PointData.GetArray('density')
	lut = lr.GetLUT(arr, 'Heat')
	rep.LookupTable = lut
	WriteImage(exportDirPath + '/' + particle_type + export_file_name + str(i) + '.png')
	Hide(cntrn)
