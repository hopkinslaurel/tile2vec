""" Based on Sherrie Wang's midwest crop script and Laurel Hopkins'
	eBird data download scripts"""


import ee
import sys
import pandas as pd

ee.Initialize()


def collect_images(locations, box_width, box_height, scale, filename, folder, export_outputs):

	# NAIP image
	naip = ee.ImageCollection('USDA/NAIP/DOQQ')\
		.filterDate('2011-01-01', '2011-12-01')\
		.reduce(ee.Reducer.median())\
		.visualize()

	for index, row in locations.iterrows():  # TODO parse row
		loc_id, lat, lon = row['LOC_ID'], row['LATITUDE'], row['LONGITUDE']
		total = locations.shape[0]

		north = lon + box_height / 2
		south = north - box_height
		east = lat + box_width / 2
		west = east - box_width

		sw_coord = ee.Geometry.Point(south, west)
		ne_coord = ee.Geometry.Point(north, east)

		try:
			polygon = ee.Geometry.Rectangle([sw_coord, ne_coord])
			export_this = naip.clip(polygon)

			poly_region = polygon.getInfo()
			poly_region = poly_region['coordinates'][0]

			task = ee.batch.Export.image.toDrive(
				image=export_this,
				description=filename + loc_id,
				folder=folder,
				fileNamePrefix=filename + loc_id,
				region=poly_region,
				scale=scale
			)

			task.start()  # could monitor statuses and submit in batches, but with 2000, shouldn't be too heavy

			if export_outputs:
				state = task.status()['state']
				while state in ['READY', 'RUNNING']:
					print state + '...'
					# print task.status()
					state = task.status()['state']

				if task.status()['state'] == 'COMPLETED':
					print 'eBird image ', str(index+1), ' of', str(total)
				else:
					print 'eBird image ', str(index+1), ' of', str(total), ' failed:'
					print task.status()
					continue
			else:
				print 'Queuing eBird image ', str(index + 1), ' of', str(total)



		except:
			print 'eBird image ' + str(index+1) + ' failed'
			print task.status()

	if export_outputs:
		print 'Finished exporting ' + str(index+1) + ' images!'
	else:
		print 'All ' + str(index+1) + ' images queued!'

if __name__ == '__main__':
	'EXPORT IMAGES'
	_locations_csv = sys.argv[1]

	'MAP PARAMETERS'
	_scale = 30  # 30 meters/pixel - native NLCD resolution -- NOT USED

	'IMAGE PARAMETERS'
	_box_width = 0.045  # TODO: TUNE
	_box_height = 0.06  # TODO: TUNE

	'EXPORT PARAMETERS'
	_folder = 'tile2vec_cluster_3bands'
	_filename = "naip_oregon_2011_cluster_"

	# enable export outputs [READY/RUNNING]
	_export_outputs = False

	# read in eBird record locations
	locations = pd.read_csv(_locations_csv)

	collect_images(locations, _box_width, _box_height, _scale, _filename, _folder, _export_outputs)
