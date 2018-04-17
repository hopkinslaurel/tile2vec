// Based on Sherrie Wang's midwest crop script

// Load a raw Landsat 5 ImageCollection for a single year.
//var collection = ee.ImageCollection('LANDSAT/LC08/C01/T1')
//    .filterDate('2013-01-01', '2013-12-31');

var collection = ee.ImageCollection('LANDSAT/LE07/C01/T1')
    .filterDate('2009-01-01', '2011-12-31');

// Create a cloud-free composite with default parameters.
var composite = ee.Algorithms.Landsat.simpleComposite(collection);

function addLatLonBands(composite){
    var ll = composite.select(0).multiply(0).add(ee.Image.pixelLonLat());
    return composite.addBands(ll.select(['longitude', 'latitude'],['PXLON', 'PXLAT']));
}
composite = addLatLonBands(composite);
composite = composite.float();
print(composite);

var box_width = 5.0;
var box_height = 5;
var north = 4;
var south = north - box_height;
var west = 30;
var east = west + box_width;
var polygon = ee.Geometry.Rectangle([west, south, east, north]);
Map.addLayer(polygon);
Map.centerObject(polygon);

Map.addLayer(composite.clip(polygon), {bands: ['B3', 'B2', 'B1'], max:68}, 'TOA composite');


var grid_size = 8;
var small_box_height = box_height / grid_size;
var small_box_width = box_width / grid_size;

var i = 0;
var j = 0;
for (i = 0; i < 8; i++) {
    var latmax = north - small_box_height * i;
    var latmin = latmax - small_box_height;
    for (j = 0; j < 8; j++) {
	var lonmin = west + small_box_width * j;
	var lonmax = lonmin + small_box_width;

	var polygon = ee.Geometry.Rectangle([lonmin, latmin, lonmax, latmax]);
	Map.addLayer(polygon);

	var export_this = composite.clip(polygon);
	Export.image.toCloudStorage({
	    image: export_this,
	    description: 'landsat7_uganda_3yr_row_'+i.toString()+'_column_'+j.toString(),
	    bucket: 'uganda_landsat',
	    fileNamePrefix: 'landsat7_uganda_3yr_row_'+i.toString()+'_column_'+j.toString(),
	    region: polygon,
	    scale: 30
	});
    }
}
