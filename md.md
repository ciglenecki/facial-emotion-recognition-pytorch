# Vislet – visualize geographic data in Memgraph Lab

# Introduction

TLDR; You have graph data that consists of nodes and edges. Nodes contain geographic data (lat, lng). Your goal is to **visualize graph data on a world map in Memgraph Lab**. 

In graph data visualization it's important to extract properties of nodes and edges and visualize them in a memorable way. Deriving new visual information from properties is a great way to make sense of your data. One of those properties can be [lat/lng coordinates](https://en.wikipedia.org/wiki/Geographic_coordinate_system#Latitude_and_longitude) of each node.

Example – dataset of neighbouring countries ([cypher](https://pastebin.com/raw/MYL1779v), [JSON](https://pastebin.com/raw/WJWiHeQ8)):
- each node is a country
- two countries are neighbors if an edge connects them

After creating the dataset in Memgraph using the cypher code above and executing a query `MATCH (n)-[r]->(m) RETURN n,r,m` in Memgraph Lab we get the following result:
![](https://i.imgur.com/5I8Wod6.png)

In Memgraph Lab visualizing any graph data is possible. With a new [style scripting language](https://) you can dynamically customize your visual output to your liking. What about geographic data? Currently, there is no proper way to make use of **lat/lng coordinates** with a style scripting language. For example, distances between countries visually and intuitively can't be shown. 

Geographic data makes sense only when you show it on a real-world map. Because of this, we created something specifically for visualizing geographic data in Memgraph Lab.

**Vislet** – a set of functions and event handlers that synchronize vis-network and Leaflet! 

![](https://i.imgur.com/eYaQw0A.png)



# How to use Vislet?

Vislet is automatically activated when it detects geographic data from returned from a query. For now, if any of the returned nodes contain **`lat` and `lng` properties** that graph data is considered geographic data. Vislet will then show nodes on the map on their respectful `lat` and `lng` positions. Nodes that don't contain `lat` and `lng` properties won't be shown on the map. In the bottom left corner you can see how many nodes are shown on the map. `Map Off` button turns the map off and shows all returned nodes, both geographic and non-geographic nodes.
![](https://i.imgur.com/Zmur6lB.jpg)
You can move and zoom on the map while nodes and edges will move and scale accordingly. Clicking on the node to get more information, expanding, collapsing and closing are features which are still available in the map view. In Vislet, node physics simulation and node dragging is turned off. Positions of nodes and edges are visually robust, as they stay in place no matter what.

For Vislet, zooming on the map isn't a problem. Visual properties like size of the node and width of the edge are scaled accordingly when zoom event occurs. Constant adjusting of style's properties values, like size and width, won't be necessary. Once you create a style, it stays intact and works on every zoom level!

![](https://i.imgur.com/aWLWz4t.png)

# How does Vislet work?

Detailed explanation behind Vislet can be found [here](link).

Vislet combines two javascript libraries to visualize graph data on a map.
- [vis-network](https://visjs.github.io/vis-network/docs/) – quickly visualize graph data. [Examples](https://visjs.github.io/vis-network/examples/) and [docs](https://visjs.github.io/vis-network/docs/network/) show you how to generate quick and good looking graphs. You can create a simple graph of neighboring countries using dataset given above.
- [Leaflet](https://leafletjs.com/index.html) – use interactive maps. Leaflet allows you to display geographic data on the map and other custom layers. [Quick start](https://leafletjs.com/examples/quick-start/) and [docs](https://leafletjs.com/reference-versions.html) both explain how to integrate a world map or a custom layer into your project. Circles, pins, markers and [more](https://leafletjs.com/reference-1.7.1.html#path) can be used to visualize data.


vis-network and Leaflet map are rendered into their own HTML div.
```html
<div id="map" style="z-index: 1"></div>
<div id="network" style="z-index: 2"></div>
```
Both vis-network and Leaflet have their own coordinate system. Vislet transforms those two coordinate systems into **one coordinate system**. Result: nodes on vis-network have a properly positioned and scaled map as a background underneath. You can still click on the nodes and edges as vis-network is above (z-index) the map. Vislet also constrains two coordinate systems as they both have their limits. Therefore, Vislet makes sure that vis-network nodes **won't**  magically **escape outside of the map**. Every geographic node can always be found on a map at it's `lat` `lng` location.
![](https://i.imgur.com/GCy0UrS.jpg)

# Next steps

Check other Vislet blogs:
- [Vislet – detailed explanation](link)
- [Vislet – full implementation](link)

Further testing and development is needed for a fully working and robust implementation of Vislet. Vislet is a new and fresh project, a lot of room for improvement. Implementation is still barebone which means every aspect of Vislet is highly customizable.
