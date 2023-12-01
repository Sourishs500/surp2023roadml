

import './App.css';
import { useLoadScript, GoogleMap, Marker } from "@react-google-maps/api";
import { useMemo } from "react";

function Square({name, text}){
  return <div class="square">
    <p id={name}>
      {text}
    </p>
  </div>
}

export default function App() {
  const { isLoaded } = useLoadScript({
    googleMapsApiKey: "AIzaSyAAo9XvOTUjeRAZQqFWErHIpPAqq6y931k",
  });
  const markers = [
    { lat: 18.5204, lng: 73.8567 },
    { lat: 18.5314, lng: 73.8446 },
    { lat: 18.5642, lng: 73.7769 },
  ];

  const onLoad = (map) => {
    const bounds = new window.google.maps.LatLngBounds();
    markers?.forEach(({ lat, lng }) => bounds.extend({ lat, lng }));
    map.fitBounds(bounds);
  };

  return (
    <>
    <div className="App">
      <head className="App-header">
        <div class="sticky">Using machine learning with passively generated data from e-scooters to analyze urban road conditions</div>   
        <Square name="initialInfo" text="Here's a map of data on routes we have currently. Click on any of the two markers to get a route and see how smooth
        your journey will be."/>
        {!isLoaded ? (
          <h1>Loading...</h1>
          ) : (
            <GoogleMap
              mapContainerClassName="map"
              onLoad={onLoad}
            >
            
              {markers.map(({ lat, lng }) => (
              <Marker position={{ lat, lng }} />
              ))}
            </GoogleMap>
            
          )}
          
      </head>

      <body>

        <div id="bottom">Made by Sourish Saswade, UCLA HiLab 2023</div> 
        
        
      </body>  
       
    </div>
    </>
  );
}



