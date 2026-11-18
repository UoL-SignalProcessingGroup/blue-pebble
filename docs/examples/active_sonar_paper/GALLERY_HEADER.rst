Active Sonar Paper
------------------

The scripts behind the active sonar paper. They share one bistatic scenario, set up in
``paper_scenario.py``: the ship, its arrays, the targets, the water and seabeds, the pulse,
the noise and the detector. The first script settles the propagation model's settings on one
ping; the later ones run the whole processing chain and the tracker over many pings.
