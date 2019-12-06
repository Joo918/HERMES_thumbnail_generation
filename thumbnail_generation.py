from paraview import lookuptable
import os
import numpy as np
import random
from string import Template
import shutil
import json
import math

###########User Input Fields##########

#particle type
particle_type = "Pi-"
#data file name
data_file_name = particle_type + "Filtered_HERMES+C.csv"
#name of the export csv. Particle type is appended as a prefix
default_export_file_name = "thumbnail"
#import file location (location of data files)
data_location = "C:/Users/Ajidot/Desktop/vt_stuff/info_viz/project/Particle_CinemaExport_ToyData"
#export file location
export_location = "C:/Users/Ajidot/Desktop/vt_stuff/info_viz/project/HERMES_thumbnail_generation/Export"
#column names to generate thumbnails by (ignore 0th index for now.)
interest_columns = [
	"Z",
	"P_hperp"
]
#data to keep for glee's parametric interaction
cols_to_preserve = [
	"Q^2",
	"x",
	"y",
	"Phi_s",
	"epsilon",
	"Z",
	"P_hperp",
	"Phi_h"
]

#DONT TOUCH
for i in range(len(cols_to_preserve)):
	if cols_to_preserve[i] == "Z":
		cols_to_preserve[i] = "Z" + "(" + particle_type[:-1] + "+/-)"
######

combination_cols = [
	"x",
	"y",
	"Z",
	"P_hperp",
	"Q^2",
	"epsilon"
]

#qzp value to filter by. If above, > 1 + qzp_filter_val / If below, < 1 - qzp_filter_val
qzp_filter_val = 0.5
#boolean. If true, filter qzp > qzp_filter_val. If false, <=
filter_above = 0

#DONT TOUCH
if filter_above:
	qzp_filter_val = 1 + qzp_filter_val
else:
	qzp_filter_val = 1 - qzp_filter_val
######

#number of thumbnails to generate
thumbnail_num = 10

x_axis_labels = [0, 0.5, 1, 1.5]
y_axis_labels = [0, 0.5, 1, 1.5, 2]
scale_arr = [1, 1, 1]
#####################################

#PROGRAMMABLE FILTER DEFINITIONS#
#####################################

#############################################################################QZP FILTER
qzp_filter = Template('''

import os
import numpy as np
import random
import json

print("filtering by qzp!")
qzp = self.GetInput().GetColumnByName('(q/zp)^2')
	
sort_col_index = 0

newlist = []
for i in range(27):
	newlist.append([])

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

minmaxFileDir = "$mMFD"
if not os.path.exists(minmaxFileDir):
	varMaxes = final_sorted_data.max(axis=1).tolist()
	varMins = final_sorted_data.min(axis=1).tolist()
	colNameArr = []
	for i in range(27):
		colNameArr.append(self.GetInput().GetColumnName(i))
	minmaxdic = {'min': varMins, 'max': varMaxes, 'names': colNameArr}
	minmax_json = json.dumps(minmaxdic)
	f = open(minmaxFileDir, 'w+')
	f.write(minmax_json)
	f.close()

for i in range(27):
	output.RowData.append(final_sorted_data[i], self.GetInput().GetColumnName(i))

''')

#############################################################################DIVN FILTER

divn_filter = Template('''
import os
import numpy as np
import math
import random

scriptDirectory = "$s_dir"
csvDirectory = "$csvd"
filenamePrefix = "$f_prefix"
ptype = "$p_type"

filename = scriptDirectory + '/tmp.txt'
#dirname = scriptDirectory + '/' + ptype + filenamePrefix
filename3 = csvDirectory + '/' + (ptype + filenamePrefix + 'thumbnails.csv').lower()
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
dat.write(',' + (ptype + filenamePrefix + str(imgnum)).lower() + '.png\\n')
dat.close()

fw = open(filename, 'w')
val = str(lastN + chunk)
fw.write(val)
fw.close()

if lastrun:
	os.remove(filename)
''')

#############################################################################DENSITY FILTER

density_filter = Template('''
import numpy as np
import os

scriptDirectory = "$s_dir"
filename = scriptDirectory + '/tmp2.txt'

if not os.path.exists(filename):
	f = open(filename, 'w+')
	f.write('999999999,-999999999')
	f.close()

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

x_arr_np = np.array(newx, float)
y_arr_np = np.array(newy, float)
output.RowData.append(x_arr_np, 'x')
output.RowData.append(y_arr_np, 'y')
densityArr = np.array(newlist, float)
dmax = densityArr.max()
dmin = densityArr.min()

fr = open(filename, 'r')
readDat = fr.read()
readDatArr = readDat.split(",")
curMin = float(readDatArr[0])
curMax = float(readDatArr[1])
fr.close()

if dmax > curMax:
	curMax = dmax
if dmin < curMin:
	curMin = dmin

fr = open(filename, 'w')
fr.write(str(curMin) + "," + str(curMax))
fr.close()

output.RowData.append(densityArr, 'density')
''')

#############################################################################DIVN FILTER

#####################################

#remove any tmp files
tmpfname = export_location + "/tmp.txt"
if os.path.exists(tmpfname):
	os.remove(tmpfname)
tmpfname2 = export_location + "/tmp2.txt"
if os.path.exists(tmpfname2):
	os.remove(tmpfname2)
	
view = GetActiveView()
#create a background image
backbox = Box()
backbox.XLength = 3
backbox.YLength = 3
backbox.ZLength = 1
backbox.Center = [1.5, 1.5, -1.0]
dpbox = GetDisplayProperties()
dpbox.DiffuseColor = [0.0, 0.0196078431372549, 0.403921568627451]
Show(backbox)
Render()

originalCamPos = view.CameraPosition
#view.CameraPosition[2] *= 0.5
#view.CameraPosition[1] -= 0.2
view.CameraPosition[0] -= 0.2
view.OrientationAxesVisibility = 0
	
qzp_dir = "a"
if not filter_above:
	qzp_dir = "b"
mainexportDirPath = export_location + "/" + particle_type + "/" + qzp_dir + "/" + particle_type.lower()

#get data
reader = CSVReader(FileName= data_location + '/' + data_file_name)
Hide()
#filter by qzp
#sort by first variable

jsonLoc = export_location+"/"+particle_type+str(filter_above)+".json"
ppf_init = ProgrammableFilter(reader)
ppf_init.Script = qzp_filter.substitute(fv=qzp_filter_val, vo=filter_above, mMFD=jsonLoc)

#set up view and heat color map
view.ViewSize = [350, 400]

lr = lookuptable.vtkPVLUTReader()
lr.Read(data_location + '/HeatColormap.xml')

##############################LOOP
for variable_x_index in range(len(combination_cols) - 1):
	for variable_y_index in range(variable_x_index + 1, len(combination_cols)):
		print("computing: ", combination_cols[variable_x_index], combination_cols[variable_y_index])
		
		interest_columns[0] = combination_cols[variable_x_index]
		interest_columns[1] = combination_cols[variable_y_index]
		
		#### RENAME Z COLS
		if interest_columns[0] == 'Z':
			interest_columns[0] = 'Z' + '(' + particle_type[:-1] + '+/-)'
		if interest_columns[1] == 'Z':
			interest_columns[1] = 'Z' + '(' + particle_type[:-1] + '+/-)'
		####

		#generate n thumbnails + export csv

		#Add x, y vars to export names
		export_file_name = default_export_file_name + "_" + interest_columns[0].replace("+/-","") + "_" + interest_columns[1].replace("+/-","")
		export_file_name = export_file_name.lower()

		#Set up export path correctly
		exportDirPath = mainexportDirPath + export_file_name
		if os.path.exists(exportDirPath):
			shutil.rmtree(exportDirPath)
		os.mkdir(exportDirPath)

		pvins = []
		for i in range(thumbnail_num):
			ppfn = ProgrammableFilter(ppf_init)
			ppfn.Script = divn_filter.substitute(s_dir=export_location, csvd=exportDirPath, f_prefix=export_file_name, p_type=particle_type, t_cols=cols_to_preserve, n=thumbnail_num, xy_cols=interest_columns)
			
			ppfdn = ProgrammableFilter(ppfn)
			ppfdn.Script = density_filter.substitute(s_dir=export_location)
			
			ttpn = TableToPoints(ppfdn)
			ttpn.KeepAllDataArrays = True
			ttpn.XColumn = 'x'
			ttpn.YColumn = 'y'
			ttpn.a2DPoints = True
			
			ttpn.UpdatePipeline()
			
			pvin = PointVolumeInterpolator(Input=ttpn, Source='Bounded Volume')
			pvin.UpdatePipeline()
			
			pvins.append(pvin)
			
			#WriteImage(exportDirPath + '/' + particle_type + export_file_name + str(i) + '.png')
			#Hide(cntrn)


		densityTmp = open(tmpfname2, 'r')
		minmaxD = densityTmp.read()
		minmaxDArr = minmaxD.split(",")
		densityTmp.close()
		os.remove(tmpfname2)

		isoMin = float(minmaxDArr[0])
		isoMax = float(minmaxDArr[1])
		isoThres = np.array(range(5), dtype=float)/4 * (isoMax - isoMin) + isoMin
		isoThresList = isoThres.tolist()
		if isoThresList[0] >= 0.00001:
			isoThresList.insert(0, 0.0)
		print("isosurface threshold = ")
		print(isoThresList)

		exportcontourlv = exportDirPath + "/contour_thresholds.txt"
		contour_f = open(exportcontourlv, 'w+')
		contour_f.write(str(isoThresList))
		contour_f.close()

		#load json to find the correct scale
		json_f = open(jsonLoc, 'r')
		json_string = json_f.read()
		json_f.close()
		parsed_json = json.loads(json_string)
		j_x_index = parsed_json["names"].index(interest_columns[0])
		j_y_index = parsed_json["names"].index(interest_columns[1])
		x_minmax = (parsed_json["min"][j_x_index], parsed_json["max"][j_x_index]) 
		y_minmax = (parsed_json["min"][j_y_index], parsed_json["max"][j_y_index])
		x_length = x_minmax[1] - x_minmax[0]
		y_length = y_minmax[1] - y_minmax[0]

		correct_x_scale = 2 * (1 / x_length)
		correct_y_scale = 2 * (1 / y_length)

		x_length *= correct_x_scale
		y_length *= correct_y_scale

		correct_x_labels = (np.array(range(5), dtype=float)/4 * (x_minmax[1] - x_minmax[0]) + x_minmax[0]) * correct_x_scale
		correct_y_labels = (np.array(range(5), dtype=float)/4 * (y_minmax[1] - y_minmax[0]) + y_minmax[0]) * correct_y_scale

		print("x labels!")
		print(correct_x_labels)
		print("y labels!")
		print(correct_y_labels)

		x_axis_labels = correct_x_labels
		y_axis_labels = correct_y_labels


		scale_arr = (correct_x_scale, correct_y_scale, 1)

		#scale_arr = (1, 1, 1)
		print("scale!")
		print(scale_arr)

		#set camera to show the correct place
		'''
		view.CameraPosition = originalCamPos
		view.CameraPosition[0] += x_length / 2
		view.CameraPosition[1] += y_length / 2
		backbox.Center = [view.CameraPosition[0], view.CameraPosition[1], -1.0]
		#maxScale = max(correct_x_scale, correct_y_scale)
		maxScale = max(x_length, y_length)
		view.CameraPosition[2] = 
		'''
		#backbox.XLength = x_length * 1.5
		#backbox.YLength = y_length * 1.5
		#backbox.ZLength = 1
		#backbox.Center = [x_length / 2, y_length / 2, -1.0]
		#Show(backbox)
		#ResetCamera()
		#view.CameraPosition[2] *= 0.7

		for i in range(len(pvins)):
			cntrn = Contour(pvins[i])
			cntrn.ComputeScalars = True
			cntrn.ComputeNormals = False
			cntrn.ComputeGradients = False
			#cntrn.Isosurfaces = [1, 1.2, 1.4, 1.6, 2]
			
			cntrn.Isosurfaces = isoThresList
			#Show(cntrs[i])
			
			rep = Show(cntrn)
			dp = GetDisplayProperties()
			dp.Representation = 'Wireframe'
			dp.LineWidth = 5
			dp.DataAxesGrid.GridAxesVisibility = 1
			dp.DataAxesGrid.XTitle = interest_columns[0].replace("+/-","") + "\nx" + '{0:.1f}'.format(correct_x_scale)
			dp.DataAxesGrid.YTitle = interest_columns[1].replace("+/-","") + "\nx" + '{0:.1f}'.format(correct_y_scale)
			
			dp.Scale = scale_arr
			
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
			
			rep.ColorArrayName = 'density'
			arr = cntrn.PointData.GetArray('density')
			lut = lr.GetLUT(arr, 'Heat')
			rep.LookupTable = lut
			
			WriteImage(exportDirPath + '/' + (particle_type + export_file_name + str(i)).lower() + '.png')
			Hide(cntrn)

print("DONE!")