import folium

# Coordinates for the Northern states in Nigeria
state_coordinates = {
    'Adamawa': (9.3266, 12.4857),
    'Bauchi': (10.3157, 9.8456),
    'Benue': (7.1956, 9.6081),
    'Borno': (11.8411, 13.1560),
    'Gombe': (10.2899, 11.1628),
    'Jigawa': (12.6769, 9.3426),
    'Kaduna': (10.3119, 7.7451),
    'Kano': (12.0022, 8.5919),
    'Kogi': (7.5664, 6.7222),
    'Katsina': (12.9954, 7.5987),
    'Kebbi': (12.4536, 4.2659),
    'Niger': (9.5736, 6.7168),
    'Plateau': (9.3251, 9.5472),
    'Sokoto': (13.0078, 5.2484),
    'Taraba': (8.8900, 11.4260),
    'Yobe': (11.4584, 11.0824),
    'Zamfara': (12.1656, 6.4964),
    'Nassarawa': (8.5685, 7.2910),
    'FCT': (9.0579, 7.49508)
}

# Initialize the map centered around Nigeria
m = folium.Map(location=[10.0, 8.0], zoom_start=6)

# Add markers for each state
for state, (lat, lon) in state_coordinates.items():
    folium.Marker(
        location=[lat, lon],
        popup=state,
        icon=folium.Icon(icon="info-sign")
    ).add_to(m)

# Save the map to an HTML file
m.save('northern_nigeria_states_map.html')
