// Based on Sherrie Wang's midwest crop script

// Load a raw Landsat 5 ImageCollection for a single year.
//var collection = ee.ImageCollection('LANDSAT/LC08/C01/T1')
//    .filterDate('2013-01-01', '2013-12-31');

var collection = ee.ImageCollection('LANDSAT/LE07/C01/T1')
    .filterDate('2009-01-01', '2011-12-31');

// Create a cloud-free composite with default parameters.
var composite = ee.Algorithms.Landsat.simpleComposite(collection);

//NAIP - 2011
var naip = ee.ImageCollection('USDA/NAIP/DOQQ')
  .filterDate('2011-01-01', '2011-12-01')
  .reduce(ee.Reducer.median())
  .visualize();

//NLCD - 2011  
var nlcd = ee.Image('USGS/NLCD/NLCD2011')
  .select('landcover');
  
var data = nlcd;

function addLatLonBands(data){
  var ll = data.select(0).multiply(0).add(ee.Image.pixelLonLat());
  return data.addBands(ll.select(['longitude', 'latitude'],['PXLON', 'PXLAT']));
}
//data = addLatLonBands(data);
//data = data.float();

print(nlcd);
print(naip);



var box_width = 7.5;
var box_height = 4.3;
var north = 46.23;
var south = north - box_height;
var west = -124.41;
var east = west + box_width;
var polygon = ee.Geometry.Rectangle([west, south, east, north]);
Map.addLayer(polygon);
Map.centerObject(polygon);

//var nlcd_reprojected = nlcd.reproject('EPSG:4326', null, 30);
//print(nlcd_reprojected);


// Reproject NLCD to match NAIP
print('projection original', nlcd.projection());
var proj = ee.Projection('EPSG:4326');
var nlcdR = nlcd.reproject(proj, null, 30)
print('projection new', nlcdR.projection());

data = nlcdR;

//Map.addLayer(data.clip(polygon), {bands: ['vis-red', 'vis-green', 'vis-blue'], min:0, max:255}, 'NAIP');
Map.addLayer(data.clip(polygon), {}, 'NLCD');

var grid_size = 20;  //8;
var small_box_height = box_height / grid_size;
var small_box_width = box_width / grid_size;

var i = 0;
var j = 0;
for (i = 0; i < 20; i++) {
  var latmax = north - small_box_height * i;
  var latmin = latmax - small_box_height;
  for (j = 0; j < 20; j++) {
    var lonmin = west + small_box_width * j;
    var lonmax = lonmin + small_box_width;

    var polygon = ee.Geometry.Rectangle([lonmin, latmin, lonmax, latmax]);
    //Map.addLayer(polygon);
    
    //Map.addLayer(data.clip(polygon), {}, 'reprojected NLCD');
    var export_this = data.clip(polygon);
    
    //print(export_this);
    //Map.addLayer(export_this);
    
    Export.image.toDrive({
      image: export_this,  //data,
      description: 'nlcd_oregon_2011_row_'+i.toString()+'_column_'+j.toString(),
      folder: 'tile2vec_10m_nlcd',
      fileNamePrefix: 'nlcd_oregon_2011_row_'+i.toString()+'_column_'+j.toString(),
      region: polygon,
      scale: 10
      });
  }
}



