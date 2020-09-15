# Vislet - Merging vis-network with Leaflet

Set of functions and event handlers that synchronize vis-network and Leaflet together.

prerequisites:

- you are using vis-network for visualizing graph data
- you are working with geographic graph data

## Motivation

In graph data visualization it's important to extract properties of nodes and edges and visualize them in a memorable way. One of those properties can be **lat/lng coordinates** of each node. 

Example -- You have a graph data of neighbouring countries ([cypher]((https://pastebin.com/raw/MYL1779v)), [JSON](https://pastebin.com/raw/WJWiHeQ8)
- each node is a country
- two countries are neighbors if an edge connects them

Visualizing this data without a map is possible. However, since we are working with geographic data it makes more sense to show it on a map.

For this, we will implement Vislet. 

before - standard vis-network graph

**image**

after - graph after implementing Vislet

**image**

TLDR: You have graph data that consists of nodes and edges. Nodes contain geographic data. The goal is to **visualize graph data on a world map** . 


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
All of 3 steps are straight forward and produce the following result:

**image**

vis-network's coordinate system can be expanded infinitely, meaning that it has no bounds. Each node is placed on the exact position given `x` and `y` properties of a node. If the node doesn't contain `x` and `y` it will be randomly placed or physics simulation will be used to determine the best visual position for the node.
![vis-networks coordinate system](https://i.imgur.com/gNSAbtn.jpg)

## Leaflet

Leaflet allows you to display geographic data on the map and other custom layers. [Quick start](https://leafletjs.com/examples/quick-start/) and [docs](https://leafletjs.com/reference-versions.html) both explain how to integrate a world map or a custom layer into your project. Circles, pins, markers and [more](https://leafletjs.com/reference-1.7.1.html#path) visual components can be used to visualize data.

If it's possible to visualize your data by using only Leaflet's API you shouldn't introduce vis-network as it might be overly complex for your project.  

Leaflet is limited to bounds of world latitude and longitude numbers. Those corners are `(90°, -180°)` and `(-90°, 180°)`. Later, this information will be used to set bounds for Vislet coordinate system.
![Leaflet coordinate system](https://i.imgur.com/fizWNFC.jpg)


## Problem

In Leaflet, it's possible to add [custom visual components](https://leafletjs.com/examples/extending/extending-2-layers.html) on the layer. Adding [HTML canvas](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API) natively on Leaflet map and drawing custom geometric objects to that canvas is [possible](https://leafletjs.com/examples/extending/extending-2-layers.html). Making objects interactive by extending existing Leaflet Layers is possible too.

The problem comes when vis-network is introduced. At first, you might want to load vis-network canvas onto Leaflet's map. vis-network canvas is accessible, but currently, there is no way to extract geometric shapes from the canvas. vis-network canvas is useless when trying to make vis-network and Leaflet work together.

## Solution

As vis-network has an infinite coordinate system we will choose Leaflet's coordinate system as our upper and lower limit. Those Leaflet limits will be defined by world map bounds and maximum map zoom. By using `map.project` function ([link](https://leafletjs.com/reference-1.7.1.html#map-project)) any geometric data can be transformed to pixel coordinate (check [Leaflet CRS origin](https://leafletjs.com/examples/crs-simple/crs-simple.html)).

This produces important results:
- each `(lat, lng)` pair can be transformed into exactly one `(x,y)` pair
- each `(x, y)` pair can be transformed into exactly one `(lat, lng)` pair

![Transformation](https://i.imgur.com/6Td1uKX.png)

What happens behind the scenes? Leaflet's `project` functions projects `[lat, lng]` pair to to pixel coordinate (check [Leaflet CRS origin](https://leafletjs.com/examples/crs-simple/crs-simple.html)). The problem is that `project` function by default takes the **current zoom** of the map. The result of this is that for `project` function will produce different pixel coordinates for a single `[lat, lng]` pair. 

To fix this, Vislet uses predefined zoom so that projection always results in the same result. Which zoom level? The **maximum map**  zoom level.

`map.project([lat, lng], map.getMaxZoom())`

Maximum zoom level allows Vislet to project `[lat, lng]` pair to pixel coordinates for vis-network even when Leaflet's map we are looking at a close street level. vis-network is therefore stretched so that it supports the placement of a node even on a street zoom level.

![](https://i.imgur.com/nOxXMkp.jpg)



1:1 mapping between Leaflet's and vis-network's is achieved. Now, any geographic point can be placed into vis-network's coordinate system.

![Vislet](https://i.imgur.com/g6rvzOC.png)





### Do it! - Implementation

Key concepts will be explained. Full code implementation is [here](https://jsfiddle.net/matejciglenecki/mboz10s9/).

The idea of implementation is as follows:
- transform Leaflet's bounds to "virtual" vis-network bounds
- assign `x` and `y` properties to each node using their respective `lat` and `lng` properties
- feed transformed nodes to vis-network

(1) - create a container for a map and vis-network
```html
<div id="map"></div>
<div id="network"></div>
```

(2) - initialize a Leaflet map. You can implement `mapOptions` by yourself. However `zoomControl` should be set to `false` as zooming and positioning will be handled by Vislet
```js
var map = L.map('map', mapOptions);
L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);
```



(3) - load geographic graph data
```js
const countryGraph = {...};
```

(3.1) - reassign property so country name is shown. There are 195 countries! It's hard to remember them all.
```js
countryGraph.nodes.forEach(node => node.label = node.properties.country_name);
```

(4) - define data for your vis-network. Pay attention to `visNodesForMap` function. This is where the magic happens. The function will assign `x`, `y` properties to each node based on their geographic location. `x` and `y` properties will be used to place nodes on correct positions when vis-network is initialized.

```js
const countryEdges = countryGraph.edges;
const countryNodes = visNodesForMap(countryGraph.nodes, map);
```
(5) - prepare a container, options and data for vis-network. Details can be found in vis-network [docs](https://visjs.github.io/vis-network/docs/network/). Important detail here is to disable dragging of nodes when network is initialized. Position of countries don't move very often! 
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

`syncPan` takes current position of the network, transforms it back into `[lat, lng]` pair and sets center of the map to those pairs.
```js
function syncPan(map, network) {
    const networkCenter = network.getViewPosition();
    networkCenter.x *= MAX_VIS_SCALE;
    networkCenter.y *= MAX_VIS_SCALE;
    const newView = map.unproject(networkCenter, map.getMaxZoom());
    map.panTo(newView, mapOptionsPan);
}
```


(8) - define Vislet `syncZoom` function. `syncZoom` synchronizes scale of vis-network and Leaflet map.

`syncZoom` takes scale of the network, transforms it back into  map zoom and sets it for the map.
```js
function syncZoom(map, network) {
    const zoom = Math.log2((network.getScale() / MAX_VIS_SCALE) * Math.pow(2, map.getMaxZoom()));
    map.setZoom(zoom, mapOptionsZoom);
}
```

(9) - set Vislet event listeners. We can listen to changes that occur whenever a position change happens on vis-network. With setting these event listeners Vislet assures that vis-network and Leaflet map are synchronized.
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

That's it, your countries are now where they should be!
