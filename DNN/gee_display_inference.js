/**
 * Display DNN land-cover inference outputs (nyvest 3-county AOI) in GEE.
 *
 * Shows the three rasters written by DNN/predict_raster.py:
 *   1. classified   — int16, raw class codes (2..12; see CLASSES below)
 *   2. uq_setsize   — uint8, LAC+Mondrian conformal set size (0..C), an
 *                     uncertainty map (1 = confident, larger = ambiguous)
 *   3. uq_pcal      — C-band uint16, per-class calibrated proba * 60000;
 *                     here shown as the WINNING-class calibrated probability
 *
 * ---- UPLOAD FIRST -------------------------------------------------------
 * Replace the placeholder asset IDs below with your uploaded assets, then run.
 * predict_raster.py writes nodata-class 0 for the classified map, 65535 for
 * uq_pcal (/60000 -> prob), 255 for uq_setsize.
 */

// ============================ ASSET IDS ==================================
var CLASSIFIED_ID = 'projects/ee-gsingh/assets/nyvest/class_2024';       // int16 class codes
var SETSIZE_ID    = 'projects/ee-gsingh/assets/nyvest/uq_2024_setsize';  // uint8 conformal set size
var MAXPCAL_ID    = 'projects/ee-gsingh/assets/nyvest/uq_2024_maxpcal';  // 1-band uint16 winning-class calibrated proba*60000
var INSET_ID      = 'projects/ee-gsingh/assets/nyvest/uq_2024_inset';    // C-band uint8 per-class 0/1 conformal-set membership
// =========================================================================

// ---- class scheme (raw codes -> name + colour) --------------------------
// Matches DNN/confusion_matrix.py LABELS. Palette must be listed in the same
// order as CLASS_CODES for the classified visualization.
var CLASSES = [
  {code: 2,  name: 'bare',      color: 'bdb76b'},
  {code: 3,  name: 'cropland',  color: 'e8d63a'},
  {code: 4,  name: 'forest',    color: '1a7d34'},
  {code: 5,  name: 'grassland', color: 'a3d977'},
  {code: 6,  name: 'scrub',     color: 'c49a52'},
  {code: 7,  name: 'wetland',   color: '5fbcd3'},
  {code: 8,  name: 'water',     color: '2b5dbd'},
  {code: 10, name: 'settle',    color: 'd93030'},
  {code: 11, name: 'sparse veg', color: 'cde6a5'},
  {code: 12, name: 'snow/ice',  color: 'ffffff'}
];

var CLASS_CODES  = CLASSES.map(function (c) { return c.code; });
var CLASS_NAMES  = CLASSES.map(function (c) { return c.name; });
var CLASS_COLORS = CLASSES.map(function (c) { return c.color; });
var NCLASS = CLASSES.length;

// ---- load rasters -------------------------------------------------------
var classified = ee.Image(CLASSIFIED_ID).selfMask();      // drop nodata (0)
var setsize    = ee.Image(SETSIZE_ID).updateMask(
                   ee.Image(SETSIZE_ID).neq(255));         // drop 255 nodata
var inset      = ee.Image(INSET_ID);                       // C-band 0/1 membership

// Winning-class calibrated probability: maxpcal is already the per-pixel max
// over classes, encoded *60000. Mask uint16 nodata (65535) before scaling.
var maxpcal = ee.Image(MAXPCAL_ID);
var pcalProb = maxpcal.updateMask(maxpcal.neq(65535))
                      .divide(60000)
                      .rename('winning_prob');

// Conformal set size derived from the inset membership bands (sum of 0/1 over
// classes) — a cross-check / alternative to the uq_2024_setsize raster.
var insetSize = inset.updateMask(inset.neq(255))
                     .reduce(ee.Reducer.sum())
                     .rename('inset_size');

// Remap raw class codes to 0..NCLASS-1 so the palette lines up exactly
// (codes are non-contiguous: 8 -> 10 skips 9).
var classViz = classified.remap(CLASS_CODES, ee.List.sequence(0, NCLASS - 1));

// ---- Sentinel-2 2024 summer cloud-free RGB composite (basemap) ----------
// Harmonized S2 SR, cloud-masked via the s2cloudless probability collection,
// median composite over the Norwegian summer (Jun–Sep) to minimise seasonal
// snow cover. Shown UNDER the inference layers.
var S2_AOI    = ee.Geometry.Rectangle([4.0, 58.5, 8.5, 63.0]);  // approx 3-county AOI
var S2_START  = '2024-06-01';   // Norwegian summer window: least snow, most sun
var S2_END    = '2024-10-01';

function maskS2clouds(img) {
  var prob = ee.Image(img.get('s2cloudless')).select('probability');
  var isCloud = prob.gt(40);
  // also drop cloud shadows via the SCL band where available (3 = shadow)
  var scl = img.select('SCL');
  var isShadow = scl.eq(3);
  return img.updateMask(isCloud.not().and(isShadow.not()))
            .divide(10000);
}

var s2sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(S2_AOI)
  .filterDate(S2_START, S2_END)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60));

var s2cloud = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterBounds(S2_AOI)
  .filterDate(S2_START, S2_END);

// join each SR scene to its s2cloudless probability image by system:index
var s2joined = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply({
  primary: s2sr,
  secondary: s2cloud,
  condition: ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
}));

var s2composite = s2joined.map(maskS2clouds).median();

// ---- map layers ---------------------------------------------------------
Map.setOptions('SATELLITE');

Map.addLayer(s2composite,
  {bands: ['B4', 'B3', 'B2'], min: 0.0, max: 0.3},
  'Sentinel-2 2024 RGB (cloud-free)');

Map.addLayer(classViz,
  {min: 0, max: NCLASS - 1, palette: CLASS_COLORS},
  'Land cover (classified)');

Map.addLayer(setsize,
  {min: 0, max: NCLASS, palette: ['000000', '1a9850', 'ffffbf', 'd73027']},
  'Conformal set size (uncertainty)', false);

Map.addLayer(pcalProb,
  {min: 0, max: 1, palette: ['440154', '31688e', '35b779', 'fde725']},
  'Winning-class calibrated prob', false);

Map.addLayer(insetSize,
  {min: 0, max: NCLASS, palette: ['000000', '1a9850', 'ffffbf', 'd73027']},
  'Set size from inset membership', false);

// centre on the AOI (approx 3-county nyvest extent, EPSG:4326)
Map.setCenter(6.4, 60.5, 8);

// ======================= LEGENDS =========================================

// -- categorical legend for the classified map --
function makeCategoricalLegend(title, names, colors) {
  var panel = ui.Panel({style: {padding: '8px', position: 'bottom-left'}});
  panel.add(ui.Label(title, {fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0'}));
  for (var i = 0; i < names.length; i++) {
    var swatch = ui.Label('', {
      backgroundColor: '#' + colors[i],
      padding: '8px', margin: '0 6px 4px 0', border: '1px solid #999'
    });
    var label = ui.Label(names[i], {margin: '0 0 4px 0', fontSize: '12px'});
    panel.add(ui.Panel([swatch, label], ui.Panel.Layout.Flow('horizontal')));
  }
  return panel;
}

// -- continuous colour-bar legend for the UQ layers --
function makeColorBar(palette) {
  // Canonical GEE gradient legend: a 100x1 image whose pixel values run 0..100
  // left-to-right (from ee.Image.pixelLonLat over the [0,100]x[0,1] bbox), so
  // the palette is sampled across its full range as a true gradient.
  var lon = ee.Image.pixelLonLat().select('longitude');
  return ui.Thumbnail({
    image: lon,
    params: {bbox: [0, 0, 100, 1], dimensions: '150x12',
             min: 0, max: 100, palette: palette},
    style: {stretch: 'horizontal', margin: '0', maxHeight: '18px'}
  });
}

function makeContinuousLegend(title, palette, tickLabels) {
  // tickLabels: array of strings spread evenly under the bar (min..max).
  var panel = ui.Panel({style: {padding: '8px', position: 'bottom-left'}});
  panel.add(ui.Label(title, {fontWeight: 'bold', fontSize: '14px', margin: '0 0 6px 0'}));
  panel.add(makeColorBar(palette));
  var ticks = ui.Panel({
    layout: ui.Panel.Layout.Flow('horizontal'),
    style: {stretch: 'horizontal'}
  });
  for (var i = 0; i < tickLabels.length; i++) {
    // first tick left-aligned, last right-aligned, middles auto-spaced
    var m = (i === 0) ? '2px 0 0 0'
          : (i === tickLabels.length - 1) ? '2px 0 0 auto'
          : '2px 0 0 auto';
    ticks.add(ui.Label(tickLabels[i], {margin: m, fontSize: '11px'}));
  }
  panel.add(ticks);
  return panel;
}

Map.add(makeCategoricalLegend('Land cover class', CLASS_NAMES, CLASS_COLORS));
Map.add(makeContinuousLegend('Set size (conf.)',
  ['000000', '1a9850', 'ffffbf', 'd73027'],
  ['0 (unknown)', '1 (confident)', String(NCLASS) + ' (unsure)']));
Map.add(makeContinuousLegend('Calibrated prob',
  ['440154', '31688e', '35b779', 'fde725'], ['0', '1']));
