# Vislet - synchronizing vis-network and Leaflet

Set of functions and event handlers that synchronize vis-network and Leaflet

prerequisites:
- you are already using vis-network for visualizing graph data
- you are working with geographic graph data

TLDR: You have graph data that consists of nodes and edges. Nodes contain geographic data (lat, lng). The goal is to **visualize graph data on a world map**. 

# Motivation

In graph data visualization it's important to extract properties of nodes and edges and visualize them in a memorable way. One of those properties can be **lat/lng coordinates** of each node. 

Example -- You have a graph data of neighbouring countries ([cypher](https://pastebin.com/raw/MYL1779v), [JSON](https://pastebin.com/raw/WJWiHeQ8))
- each node is a country
- two countries are neighbors if an edge connects them

Visualizing this data without a map is possible. However, since we are working with geographic data it makes more sense to show it on a map. For this, we will implement Vislet!


![](https://i.imgur.com/3zK3jog.jpg)
before - a standard vis-network graph

![](https://i.imgur.com/MBD7Grz.jpg)
**after - successful Vislet implementation** 


## vis-network
vis-network allows you to quickly visualize graph data. [Examples](https://visjs.github.io/vis-network/examples/) and [docs](https://visjs.github.io/vis-network/docs/network/) show you how to generate quick and good looking graphs. You can create a simple graph of neighboring countries using dataset given above. Before vis-network initializations 3 parameters must be set: 
1. container
2. options
3. data

```
const countryGraph = {...} // neighbouring countries data

const countryEdges = countryGraph.edges;
const countryNodes = countryGraph.nodes;

const visContainer = document.getElementById("network");
const visOptions = {};
const visData = {
    nodes: countryNodes,
    edges: countryEdges,
}

const network = new vis.Network(visContainer, visData, visOptions);
```
All of 3 parameters are straight forward and result in a vis-network graph on shown on the first picture.

vis-network's **coordinate system** can be **expanded infinitely**. It has no bounds. Each node is placed on the exact position given `x` and `y` properties of a node. If the node doesn't contain `x` and `y` it will be randomly placed or vis-network's physics simulation will be used to determine the best visual position for the node.

![vis-networks coordinate system](https://i.imgur.com/gNSAbtn.jpg)

## Leaflet

Leaflet allows you to display geographic data on the map and other custom layers. [Quick start](https://leafletjs.com/examples/quick-start/) and [docs](https://leafletjs.com/reference-versions.html) both explain how to integrate a world map or a custom layer into your project. Circles, pins, markers and [more](https://leafletjs.com/reference-1.7.1.html#path) visual components can be used to visualize data.

![](https://i.imgur.com/YOBlfPh.jpg)

If it's possible to visualize your data by using Leaflet's API try avoiding vis-network. Introducing complexity might not be necessary foryour project!

Leaflet is limited to world bounds defined by latitude and longitude. Those limits are corners: `(90°, -180°)` and `(-90°, 180°)`. Any point in the world can be defined using a proper latitude and longitude pair. Because bounds of the world are limited by latitude and longitude, Vislet will later use these latitudes and longitudes to limit vis-networks infinite coordinate system.
![Leaflet coordinate system](https://i.imgur.com/fizWNFC.jpg)


# Problem

In Leaflet, it's possible to add [custom visual components](https://leafletjs.com/examples/extending/extending-2-layers.html) on Leaflet's map. Already mentioned geometric shapes + more complex shapes can be added onto a [HTML canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API). Once all shapes are added to a HTML canvas, that canvas can be **natively**  added on a Leaflet map ([link](https://leafletjs.com/examples/extending/extending-2-layers.html)). You can make those geometric objects interactive. They can become clickable and show useful information (for that you have to extend existing Leaflet Layers).

The problem comes when **vis-network** is introduced. You are already familiar with using vis-network. You created a beautiful visual look for your network and you want to keep your visual progress without introducing new libraries.

It might be tempting to use vis-network's canvas and render it onto a Leaflet's map. That's right, **vis-network canvas** is accessible and can be found in `network`'s properties. We have a HTML canvas, what now? **Dead end**. Currently, there is no way to extract geometric shapes from the canvas. vis-network also doesn't contain proper geometric (visual) information which can be used to redraw geometric shapes. vis-network canvas is useless when trying to make vis-network and Leaflet work together.

# Solution

Show your vis-network data on the map! **Implement Vislet!** Vis allows you use vis-network with Leaflet maps. Vislet respects geographical properties of nodes and correctly renders them on a Leaflet map. How does Vislet work?

vis-network has an infinite coordinate system, Vislet chose Leaflet's coordinate system it's limiting/bounding coordinate system. Again, Leaflet limits are defined by world map bounds `(90°, -180°)` and `(-90°, 180°)`. With that in mind, everything that happens on the vis-network must be seen on the Leaflet map. Vislet makes sure that vis-network nodes **won't**  magically **escape outside of the map**. This limitation is set in Vislet. Vislet implements Leaflet [function](https://leafletjs.com/reference-1.7.1.html#map-project) `map.project` in a specific way and enforces this useful limitation.

`map.project` is a function that transforms any geographical data into pixel coordinates (check [Leaflet CRS origin](https://leafletjs.com/examples/crs-simple/crs-simple.html)).

`map.project((lat, lng), map.getMaxZoom())`



Transformation produces important results:
- each `(lat, lng)` pair can be transformed into exactly one `(x,y)` pair
- each `(x, y)` pair can be transformed into exactly one `(lat, lng)` pair

![Transformation](https://i.imgur.com/6Td1uKX.png)



What happens behind the scenes?
> Leaflet's `map.project` function projects `(lat, lng)` pair to `(x, y)` pair (check [Leaflet CRS origin](https://leafletjs.com/examples/crs-simple/crs-simple.html)). The problem is that `map.project` function can return different `(x, y)` pair for exactly the same `(lat, lng)` pair! Why? For its projection, `map.project` uses a **given zoom level** as an argument. By default, this argument is set to **map's current zoom**. Whenever we zoom in on the map, to more closely check out some town, map's current zoom changes. With that, projection of `(lat, lng)` pair will produce a different result, a different `(x, y)` pair. This is bad. Projectional inconsistency between `(lat, lng)` pair and `(x, y)` pair must be solved.

> In Vislet, this problem is fixed. Vislet can project `(lat, lng)` pair to `(x, y)` pair **independently of the map's current zoom level**. Using Leaflet on a continent or a street zoom level is the same. Behind the scenes, Vislet will always project data accordingly. This is because Vislet sends **maximum map zoom** as an argument for projection. The maximum zoom level is usually defined before map's initialization. By using maximum map zoom in `map.project` function, `(lat, lng)` pair always returns the same `(x, y)` pair and vice versa. **1:1 mapping between Leaflet map and vis-network is achieved.** 
![](https://i.imgur.com/ss3192j.png)


Edge cases for transformation: 
- minimum `(90°, -180°)` ==map.project==> (0, 0)
- maximum `(-90°, 180°)` ==map.project==> $(2^{maxZoom}\cdot256, 2^{maxZoom}\cdot256,)$

Examples for max zoom level 7:
  - `(-35°, -120°)` ==map.project==> `(5461, 19788)`
  - `(-10°, 10°)` ==map.project==> `(17294, 17298)`
  -  `(-90°, 180°)` ==map.project==> `(32678, 32678)`


(Optional) - Why the maximum map zoom level? 
> The answer lies in vis-network. vis-network can be infinitely expanded and you can zoom out as much as you want. However, vis-network has an upper zoom limit of 10 (this has nothing to do with map's zoom level). In vis-network you can't zoom in infinitely.

> Imagine this, you are zooming in on your vis-network and Leaflet's map zooms accordingly. However, next zoom won't happen because vis-network's zoom is currently 10. You've hit the upper limit. You can't zoom further. Your view would be locked on a country scale but you want to check out the streets too. You simply can't.

> Vislet's solution is to bring vis-network's upper zoom limit to the street level (maximum zoom level). vis-network's scale depends on Leaflet map's maximum zoom and vis-network is scaled in such a way it can't cross upper zoom limit of 10. **Vis-network's max zoom limit (10) is achieved only when Leaflet's max zoom level (user defined) is achieved.** This works for any Leaflet's max zoom level.

> This way, you can zoom in and out of the map as much as you want. Nodes will scale accordingly all the way because the upper limit (zoom in) of vis-network isn't possible to reach and the lower limit (zoom out) doesn't exist.

Finally, Vislet will assign pixel coordinates (`(x, y)` pairs) for each node by projecting their `(lat, lng)` properties and using map's maximum zoom level. With `x` and `y` properties, Vislet can show nodes directly on the vis-network. Any geographic point can be transformed and placed into vis-network's coordinate system.

![Vislet](https://i.imgur.com/g6rvzOC.png)


# Do it! - Implementation

Check full implementation [here](https://jsfiddle.net/matejciglenecki/mboz10s9/).

Rough Vislet implementation is as follows:
- define proper projections and unprojections
- assign `x` and `y` properties to each node using their respective `lat` and `lng` properties
- initialize vis-network with transformed nodes 
- keep vis-network and Leaflet in sync

**The following steps are not enough for full working application.**  These steps are key concepts of implementing Vislet:

(1) - create a container for Leaflet map and vis-network. Map and network are two separate entities.
```html
<div id="map"></div>
<div id="network"></div>
```

(2) - initialize a Leaflet map. You implement `mapOptions` yourself. To avoid accidental zooms, set `zoomControl: false` as zooming and positioning will be handled by Vislet
```js
var map = L.map('map', mapOptions);
L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);
```



(3) - load geographic graph data with nodes that contain lat and lng properties
```js
const countryGraph = {...};
```

(Optional 3.1) - vis-network can show node's label. Reassign property `label` so country name is shown. There are 195 countries, it's hard to remember them all!
```js
countryGraph.nodes.forEach(node => node.label = node.properties.country_name);
```

(4) - transform data for your vis-network. Pay attention to `visNodesForMap` function. This is where the magic happens. The function will assign `x`, `y` properties to each node based on the map's maximum zoom and node's `lat` and `lng` propeties. `x` and `y` properties will be used to place nodes on correct positions when vis-network is initialized.

```js
const countryEdges = countryGraph.edges;
const countryNodes = visNodesForMap(countryGraph.nodes, map);
```

(5) - prepare a container, options and data for vis-network. Details can be found in vis-network [docs](https://visjs.github.io/vis-network/docs/network/). Important detail here is to disable dragging of nodes when network is initialized. Position of countries don't change very often! 
```js
var visContainer = document.getElementById("network");

const visOptions = {
    autoResize: false,
    edges: {
        smooth: false,
    },
    physics: false,
    interaction: {
        dragNodes: false
    },
};

const visData = {
    nodes: countryNodes,
    edges: countryEdges,
}
```

(6) - create a vis-network
```js
var network = new vis.Network(visContainer, visData, visOptions);
```

(7) - define Vislet `syncPan` function. `syncPan` synchronizes position of vis-network and Leaflet map. 

`syncPan` takes current position of the network, transforms it back into `(lat, lng)` pair and sets center of the map to those pairs.
```js
function syncPan(map, network) {
    const networkCenter = network.getViewPosition();
    networkCenter.x *= MAX_VIS_SCALE;
    networkCenter.y *= MAX_VIS_SCALE;
    const newView = map.unproject(networkCenter,5);
    map.panTo(newView, mapOptionsPan);
}
```


(8) - define Vislet `syncZoom` function. `syncZoom` synchronizes scale of vis-network and Leaflet map.

`syncZoom` takes scale of the network, transforms it back into  map zoom and sets it for the map.
```js
function syncZoom(map, network) {
    const zoom = Math.log2((network.getScale() / MAX_VIS_SCALE) * Math.pow(2,5));
    map.setZoom(zoom, mapOptionsZoom);
}
```

(9) - set Vislet event listeners. We can listen to changes that occur whenever view is changed on vis-network. Setting these Vislet listeners assures that vis-network and Leaflet map in sync.

Note: `syncView` = `syncPan` + `syncZoom`
```js
function setVisletListeners(network, map) {
    network.once('afterDrawing', () => {
        setInitialView(network, map)
        syncView(map, network);
    });

    network.on('zoom', () => {
        syncView(map, network);
    });

    network.on('dragStart', () => {
        syncPan(map, network);
    });

    network.on('dragging', () => {
        syncPan(map, network);
    });

    network.on('dragEnd', () => {
        syncPan(map, network);
    });
}

setVisletListeners(network, map)
```

(10) - load the rest of Vislet-util functions found in the full implementation



![](https://i.imgur.com/MBD7Grz.jpg)
Countries are now in their place!


Further testing is needed for a fully working and robust implementation of Vislet. This is a new and fresh project, a lot of room for improvement. Implementation is still barebone which means every aspect of Vislet is highly customizable. I invite you to check out the full implementation [here](https://jsfiddle.net/matejciglenecki/mboz10s9/) and come up with something that will help others visualize vis-network data on Leaflet maps!
