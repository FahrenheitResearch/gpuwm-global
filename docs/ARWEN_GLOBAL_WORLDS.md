# Other worlds

**Experimental.** Everything on this page is a research tool. Its keys,
labels and limits may change in any release, and nothing here is a forecast
of anything.

Arwen Global can run Earth changed: its mountains lowered or removed, its axis
tilted, its sun brighter or dimmer, its air richer or poorer in carbon dioxide,
its sea raised, its land taken away, or a continent from a map that is not
today's. A world is a few lines in the ordinary run config. It has its own
identity, it is checked on the CPU before any card time, and it is named on
every receipt, checkpoint and tape it writes.

Every result of a world carries one sentence, and it is the first thing to
know about all of them:

> This is the weather of the first N days after Earth was changed, over a sea
> surface that does not respond. It is not this world's climate.

The sea surface temperature is held for the whole run. Nothing in this package
yet lets a sea or its ice answer the atmosphere, so a world here is weather
over a sea somebody prescribed, never the climate that world would settle
into.

The measurements on this page were taken on the CPU, except the two forecast
days of the shipped worlds, which name their card. Each one names the test
that produced it, or the frame and the date it was read from.

---

## Three labels

Every world gets one of three labels, printed on its statement and its tapes:

| Label | What it means |
|---|---|
| **science** | a result that could be defended against the literature |
| **illustration** | plausible and self-consistent enough to show; not evidence about that world |
| **not supported yet** | refused before any card time, with the reason |

Earth with its mountains lowered or removed is labelled **science for days 2
to 6 near the ranges, illustration elsewhere**: days 0 to 2 are the air
falling into the space where the rock was, and after about a week a
difference far from the ranges cannot be told from two Earth forecasts
drifting apart on their own. Every other world that runs is an
**illustration**. A Pangea is an illustration: Earth's air over a
Pangea-shaped continent, not the Permian.

A label can only get worse during a run, never better. The rule that gives
each world its label is in `arwen_global.world.statement`, and
`tests/test_arwen_global_world_preflight.py` holds it to the labels above.

---

## Writing a world

A world is a `[world]` table at the end of any analysis-start config. Every
key is optional, and a key left out is Earth's:

```toml
[world]
label = "flat-earth"           # printed on every receipt, tape and picture

[world.terrain]
scale = 0.0                    # 0 = no mountains, 1 = Earth
air_mass = "conserve"          # keep Earth's amount of air; "fill" keeps sea-level pressure

[world.sun]
obliquity_deg = 23.5
brightness = 1.0               # times the model's own solar constant
day_length_h = 24.0            # the rotation rate follows it

[world.sea]
level_m = 0.0
land = "keep"                  # "none" = waterworld
temperature = "analysis"       # or "zonal-mean", or a published profile
ice = "analysis"               # or "zonal", "none"

[world.air]
co2_ppm = "dated"              # or a number
```

Every key the table accepts, with Earth's value and the range the parser
holds it to. A value outside its range is refused by its key's name.

| Key | Earth | Accepted |
|---|---|---|
| `label` | none | lower-case letters, digits, `.`, `_` and `-`, up to 48 characters |
| `terrain.scale` | 1.0 | 0 to 2 |
| `terrain.air_mass` | `"conserve"` | `"conserve"`, `"fill"` |
| `terrain.cap_m` | none | 0 to 9,000 m: no terrain higher than this |
| `terrain.box.south_deg`, `north_deg`, `west_deg`, `east_deg` | none | the edit acts only inside the box; a box may cross the date line |
| `terrain.box.taper_deg` | none | 0 to 30 degrees, back to Earth's terrain over this distance |
| `sun.obliquity_deg` | 23.5 | 0 to 180 |
| `sun.brightness` | 1.0 | 0.1 to 4 |
| `sun.day_length_h` | 24.0 | 12 to 2,400 |
| `sun.orbit` | `"earth"` | `"earth"`, `"perpetual-equinox"`, `"eccentric"` |
| `sun.eccentricity`, `sun.perihelion_deg` | none | an eccentric orbit only; eccentricity 0 to 0.3 |
| `sea.level_m` | 0.0 | 0 to 200 m |
| `sea.land` | `"keep"` | `"keep"`, `"none"` |
| `sea.temperature` | `"analysis"` | `"analysis"`, `"zonal-mean"`, `"aqua-control"`, `"aqua-peaked"`, `"aqua-flat"`, `"aqua-qobs"` |
| `sea.ice` | `"analysis"` | `"analysis"`, `"zonal"`, `"none"` |
| `air.co2_ppm` | `"dated"` | `"dated"` or a number above 0 and under 10,000 |
| `air.ch4_ppb`, `air.n2o_ppb` | `"dated"` | `"dated"` or a number above 0 and under 100,000 |
| `surface.folder` | none | an invented world's folder, by name |
| `surface.land_cover` | none | `"open-shrubland"`, `"barren"` |
| `surface.soil_wetness` | none | 0 to 1, a fraction of saturation; required with a folder |

The names are fixed. No key under `[world]` is called `name`, `duration_s`,
`output_interval_s`, `analysis_grib`, `start_time_utc`, `analysis_mapping` or
`radiation_column_chunk`, because a launcher that sets those by pattern needs
each to occur once in the file. `tests/test_arwen_global_world_spec.py` holds
the names, the ranges and every refusal's text.

Five settings are refused by name with the reason, because the model cannot
yet honour them: another gravity, another radius, another year length,
another air composition, and another amount of air.

### Identity

A world whose every setting is Earth's has the identity `none` and joins
nothing. A config with no `[world]` table and a config with an all-Earth one
hash the same, and ten steps of each write the same bytes; this is the gate
the worlds work keeps on every shipped experiment
(`tests/test_arwen_global_world_spec.py`,
`test_every_shipped_config_hashes_the_same_with_an_all_earth_world` and
`test_ten_steps_with_an_all_earth_world_are_the_bytes_of_ten_without`).

Any other world's identity is the SHA-256 of its settings. The label is a name
and is left out. The identity joins the config hash, the receipt, every
checkpoint's metadata and, for a world with its own surface, the statics cache
name. A restart from a checkpoint of another world is refused with the named
reason, and a checkpoint or receipt written before worlds existed reads as
Earth (`tests/test_arwen_global_world_downstream.py`).

---

## Starting one

The shipped presets are listed by `gpuwm-global world list`, and
`gpuwm-global world new` writes a config from an Earth one with a preset's
table added:

```bash
gpuwm-global world new flat-earth \
    --from arwen_global_gdas_t255_native_sl_si_24h --out flat-earth.toml
```

| Preset | The world | Label |
|---|---|---|
| `flat-earth` | Earth with no mountains, Earth's amount of air | science for days 2 to 6 near the ranges; illustration elsewhere |
| `half-mountains` | every mountain at half its height | science for days 2 to 6 near the ranges; illustration elsewhere |
| `no-tibet` | no Tibetan plateau or Himalaya, the rest of Earth kept | science for days 2 to 6 near the ranges; illustration elsewhere |
| `no-rockies` | no Rocky Mountains, the rest of Earth kept | science for days 2 to 6 near the ranges; illustration elsewhere |
| `no-seasons` | no tilt: the sun stays over the equator all year | illustration |
| `tilt-54` | a 54 degree tilt | illustration |
| `on-its-side` | a 90 degree tilt, Earth on its side | illustration |
| `dim-sun` | a sun 6 percent dimmer | illustration |
| `bright-sun` | a sun 6 percent brighter | illustration |
| `ice-age-air` | glacial CO2, 190 ppm | illustration |
| `hothouse-2x` | CO2 at 560 ppm | illustration |
| `hothouse-4x` | CO2 at 1120 ppm | illustration |
| `flood-70m` | the sea 70 m higher over today's land | illustration |
| `waterworld` | no land, flat, Earth's zonal-mean sea | illustration |

Two worlds also ship as complete experiments, each the T255 forecast-day
experiment `arwen_global_gdas_t255_native_sl_si_24h` with one `[world]` table
added and nothing else changed: `arwen_global_world_t255_no_mountains` and
`arwen_global_world_t255_on_its_side`. They resolve by bare name like every
shipped experiment (`tests/test_arwen_global_world_templates.py`). Each ran
its forecast day from its GDAS 2026-09-01 00Z analysis on one RTX 5090 on
2026-11-12, beside the plain experiment, and passed every gate: 368 s for
no mountains, 364 s on its side, 376 s for the plain experiment. The no
mountains start's lowest surface pressure was 92,401 Pa (the plain
experiment's 51,354 Pa) with the global mean within 0.9 Pa of the plain
experiment's, so Earth's amount of air.

An invented world (Pangea, a painted map) is not a preset: it is a folder the
user builds, described under [Invented worlds](#invented-worlds) below.

### Before any card time

`gpuwm-global world describe CONFIG` prints what the world changes, its
identity and the one sentence. The full world statement adds what was
invented and by which rule, what was held, the label, and four margins
computed for this world from the analysis it will start from:

1. the hottest land skin expected, against the radiation's 355 K limit;
2. the coldest air expected, against the dynamics' 140 K limit (the radiation
   treats air colder than 160 K as 160 K);
3. the days until the first ocean column empties the 500 kg m-2 water store
   every column pays its evaporation from, against the run's length;
4. the bottom layer's thickness at the lowest surface pressure.

The same check refuses a start the model is known not to survive, by name and
before the cold start reads a field:

| Refusal | When |
|---|---|
| `terrain_too_tall` | the predicted lowest surface pressure is under 45,000 Pa |
| `pressure_over_ceiling` | the predicted surface pressure is over 115,000 Pa, the deepest layer the radiation accepts (its gas tables end at 109,663 Pa and a layer between the two is looked up at their deepest pressure) |
| `start_skin_over_limit` | a land skin, warmed 6.5 K per km of height removed, passes 355 K |

Each limit is a row with the code that enforces it, in
`arwen_global.world.preflight`; `tests/test_arwen_global_world_preflight.py`
fires every refusal and checks its text. The Earth figures the margins start
from were read from this model's own T255 output: the hottest skin 336.58 K
(2026-11-11 03Z, over Australia, from GDAS 2026-11-10 18Z), the coldest air
175.17 K (hour 72 of a 5-day run from ERA5 2019-01-15), and 23.9 days until
the first ocean column empties its store (a 5-day run from ERA5 2019-01-15).
None of them is a northern-summer reading, so the hot margin of a northern
summer start is not yet bounded by a measurement.

---

## What each part of a world does

### Mountains

The analysis is edited after it is decoded and before the unchanged cold start
reads it. The new terrain is `scale` times Earth's, capped at `cap_m`, inside
the box and tapered back to Earth's outside it.

Surface pressure is moved to the new ground layer by layer through the
column's own temperature and humidity. Air under removed ground follows one
rule for every analysis centre: its temperature continues downward at 6.5 K
per km from the lowest level that was above ground, and its specific humidity
is held at that level's value, so a flattened plateau starts as a hot, dry
low and no water is added that Earth did not have. `conserve` then scales the
surface pressure everywhere by one number, so the world keeps Earth's amount
of air and its whole pressure map sits uniformly lower than Earth's; `fill`
keeps the sea-level pressure map and adds air. Land skin and soil warm by
6.5 K per km of height removed, and snow is taken off where the warmed skin is
above freezing.

`tests/test_arwen_global_world_frame_edits.py` holds the mover to the closed
form of a constant-lapse atmosphere lowered 500, 2,000 and 4,500 m, returns
an unchanged column bit for bit, and returns a column lowered and raised back
to within 1 Pa.

**Measured on a real frame, 2026-11-12,** by the preflight on the public GDAS
analysis of 2026-11-10 18Z at 0.25 degrees, the frame's own grid.
`tests/test_arwen_global_world_preflight.py` carries a real-frame check that
runs when `ARWEN_WORLD_GDAS_FRAME` names a GDAS analysis: it asserts the first
refusal below and prints the statements of mountains at 1.5 times their
height and of no mountains under `conserve`.

- Mountains at 1.5 times their height are refused: the lowest predicted
  surface pressure is 32,987 Pa over the Himalaya, and over the Dead Sea the
  highest is 109,678 Pa.
- Earth with no mountains under `conserve` starts between 92,528 and
  103,386 Pa with its mean surface pressure unchanged at 98,568 Pa. Under
  `fill` it reaches 106,063 Pa and holds 2.59 percent more air.
- **That frame cannot start a world without mountains.** At 18Z it is early
  afternoon on the Altiplano, where the analysis skin is 330.67 K at 4,013 m.
  Warmed by 6.5 K per km it would be 356.75 K, past the radiation's 355 K,
  and the start is refused as `start_skin_over_limit`. A start from an
  analysis at which South America is dark does not meet this.

### The sun

A world's sun is the model's own radiation with its tilt and brightness
changed and nothing else: a subclass that overrides the two methods holding
the sun, built only when the sun is not Earth's. At Earth's values it returns
the parent's arrays bit for bit, its global-mean sunlight is a quarter of the
solar constant, and its annual mean at the pole is the solar constant times
the sine of the tilt over pi (`tests/test_arwen_global_world_sun.py`, which
also pins the names of the two parent methods, so a change to the carried
radiation cannot silently stop a world's sun from applying).

Ozone, vegetation and the land-use season stay on Earth's calendar. The world
statement prints the hot and cold margins of a tilt above Earth's as not
bounded rather than invent a number, and above 45 degrees with land it warns
that a summer continent can pass the radiation's 355 K skin limit within a
season. That is an estimate, not a measurement: no world with a large tilt
has been run long enough to read it, and if it happens the run stops by name.

### The day

A world's day turns the sun's hour angle and spins the dynamics from one
function, so the two cannot disagree. The time on every tape stays the
ordinary calendar time, the start plus elapsed seconds; on a world whose day
is not 24 hours that label no longer says where the sun is, so each tape also
carries the true sub-solar longitude and declination (an eccentric orbit
writes none yet, and says so). The two instruments
that read local solar time, the diurnal-phase and trigger diagnostics, refuse
a run whose day is not 24 hours (`tests/test_arwen_global_world_clock.py`).
The shortest day the parser accepts is 12 hours, because the stability of the
dynamics' rotation term at faster rotation has not been measured.

### The sea and its ice

The sea temperature is held for the whole run and comes from one of three
sources: the analysis as it is (a column the world floods takes its row's
value), the zonal mean of the analysis's own open water by latitude, or one
of the four aquaplanet profiles of Neale and Hoskins. With phi the latitude,
each is 27 C at the equator and 0 C at and beyond 60 degrees:

| `sea.temperature` | Inside 60 degrees, in C |
|---|---|
| `aqua-control` | 27 (1 - sin²(3 phi / 2)) |
| `aqua-peaked` | 27 (1 - 3 abs(phi) / pi) |
| `aqua-flat` | 27 (1 - sin⁴(3 phi / 2)) |
| `aqua-qobs` | the mean of control and flat |

The ice is kept, made zonal (each row's mean fraction), or removed
(`tests/test_arwen_global_world_sea.py`). No ice grows or melts.

On this line every column, ocean included, pays its evaporation from a finite
store of 500 kg m-2, and a column that empties stops the run. At the rates
read from the run named above, the first ocean column would empty its store
after 23.9 days, so the world statement compares every run's length with that
figure and says how close it comes.

### The coast

A world that raises the sea or removes the land edits Earth's static surface
in memory, inside the one reader every caller loads it through, so the cold
start, the land surface and the sea all see the same coast. Raising the sea
turns exactly the land at or below the new level into sea. There is never a
file on disk that looks like an Earth cache and is not one
(`tests/test_arwen_global_world_statics.py`).

The model's own renderer draws a world's coast from the outline of its tapes'
land mask, never Earth's lines, and puts the world's label on every panel
title (`tests/test_arwen_global_world_coast.py`). A world with no land draws
no coast.

### Invented worlds

An invented world is a folder holding the user's map and everything built
from it, named in `[world.surface] folder`. The open PALEOMAP elevation maps
are the first source: they are fetched on request into their own empty
folder, checked against the record's size and checksum, and opened as data.
Nothing is bundled.

```bash
python -m arwen_global.world.sources.paleodem fetch maps
python -m arwen_global.world.sources.paleodem describe maps --age 250
python -m arwen_global.world.sources.paleodem make-world maps --age 250 --world pangea-250ma.world
python -m arwen_global.world.surface pangea.toml
```

The second line prints which map was chosen and from what. The last builds the
world's static surface on the model grid from the config that names the
folder. The rules are deliberately plain and each is written beside the built
surface:

- the sea is every point at or below sea level that is joined to the largest
  ocean, so a basin cut off from it is dry land and the world has no inland
  lakes;
- every land column has one land-cover class (open shrubland, or barren) and
  one soil (loam), with vegetation and albedos from the WRF tables for that
  class, the same every month;
- no land ice;
- the deep soil is Earth's zonal mean at sea level, lapsed to the world's
  terrain;
- the air is Earth's zonal mean, laid onto the world's terrain by the same
  layer-by-layer pressure mover, and the starting soil wetness is the number
  the user types.

The terrain the dynamics see is the map averaged to the model's cells and
spectrally filtered, and the build prints the largest false height the filter
leaves over open ocean (`tests/test_arwen_global_world_surface.py`,
`tests/test_arwen_global_world_terrain.py`,
`tests/test_arwen_global_world_frame.py`).

**What the maps hold, read 2026-11-12** from the 1 degree archive the reader
fetches: 109 maps from 0 to 540 million years ago, one elevation variable
named `z` with no unit attribute, on a 181 by 361 point grid whose last column
repeats the first. The age in a file name is nominal: in 83 of the 109 maps
the file's own history names an age up to 3.2 million years away, and the
reader prints both. The 250 million year map spans -3,800 to 2,320 m and has
land on 28.7 percent of the globe. The present-day map's land agrees with
this model's own T255 Earth land mask on 97.0 percent of the globe, and the
largest part of the rest is Antarctica, where that map carries the rock under
the ice sheet, below sea level, rather than the ice surface. The comparison is
`python -m arwen_global.world.sources.paleodem compare` with an Earth statics
cache, and `tests/test_arwen_global_world_paleodem.py` holds the reader on
small synthetic maps.

The starting soil wetness of an invented world decides the interior of its
continent for weeks. Run two wetnesses side by side before reading anything
into one.

---

## Twins and the noise floor

Two forecasts from the same analysis drift apart on their own, so one world
beside one Earth forecast is not a measurement after about a week. Every twin
is therefore three runs from one analysis:

1. the control: the world's config with its `[world]` table taken out;
2. the perturbed control: the control restarted from its own first checkpoint
   with its temperature moved at rounding level, the same bytes for the same
   seed;
3. the world.

`gpuwm-global world twin CONFIG` writes the control's config and prints the
commands of the three runs. `gpuwm-global world diff` writes difference tapes
that show world minus control only where it exceeds perturbed control minus
control, and removes and prints the uniform pressure offset that
`air_mass = "conserve"` leaves over unchanged ground. A twin of an all-Earth
world differs by exactly zero (`tests/test_arwen_global_world_twin.py`).

Surface pressure itself is shown only where the ground did not move by more
than 1 m. Earth's spectral terrain ripples by more than that over almost all
of the sea, so for a world with no mountains the surface-pressure difference
covers a few percent of the globe; the sea-level pressure difference is the
one to draw.

---

## The demo worlds at T127

Seven of the worlds ran on one RTX 5090 on 2026-11-12 and 13, each on the
T255 forecast-day experiment `arwen_global_gdas_t255_native_sl_si_24h` cut to
truncation 127 (`dt_s = 600`, `land_surface_interval_s = 600`,
`dx_m = 104200`) with the world's table appended, from the public GDAS
analyses of 2026-09-01 00Z and 2026-11-10 18Z. Every run finished its length
and passed every gate on its receipt. Over each run the global-mean surface
pressure moved by at most 1.2e-6 of itself and the global-mean total water the
run books (`global_mean_total_water_kg_m2`) did not move in its fourth
decimal (numbers from each run's `diagnostics.jsonl`). The no-mountains and
on-its-side worlds each ran beside their Earth control and a perturbed
control (`gpuwm-global world perturb`, seed 1), and their pictures were drawn
from `gpuwm-global world diff`. Wall times include the cold start and every
checkpoint.

| World | Length | Wall | What it showed | Label |
|---|---|---|---|---|
| Earth (control) | 20 days | 1,921 s | the reference | |
| `flat-earth` | 10 days | 1,053 s | see below | science near the lowered ranges, days 2 to 6; illustration elsewhere |
| `on-its-side`, from 2026-11-10 18Z | 5 days | 373 s | sun over 50.5 S at the start, 55.5 S at day 5 (tape attributes); land 2 m temperature minus Earth's at day 5, band means: 30 to 60 N -6.3 K, 0 to 30 N -12.0 K, 30 to 60 S +4.5 K, 60 to 90 S +17.3 K, against a perturbed-control noise of 0.55 K (global RMS) | illustration |
| `waterworld` | 20 days | 1,609 s | no land and no coast on any tape; rain over days 10 to 20 peaks at 9.9 mm a day in one band from 10 to 20 N, global mean 3.9 mm a day against Earth's 3.0 | illustration |
| a PALEOMAP 250 Ma world folder, soil wetness 0.25 and 0.75 | 20 days each | 1,422 s and 1,345 s | its own coast drawn from the tape's land mask; mean land 2 m temperature of the dry start 0.45 K above the wet one at day 20, and land rain 1.48 against 1.80 mm a day since the start | illustration |
| `flood-70m` | 10 days | 738 s | 1,712 land columns of 24,880 become sea, Florida's 12 among them; 2 m temperature minus Earth's at day 10: +0.82 K over the flooded columns, -0.06 K over the land that stayed | illustration |
| a 12-hour day | 20 days | 1,506 s | the sub-solar point is back over the date line every 12 hours of the clock; the zonal wind at 250 hPa starts at half of Earth's (11.3 against 22.1 m/s in the north), and the 500 hPa height's mean zonal wavenumber at 30 to 60 degrees is 5.0 to 6.7 over days 5 to 20 against Earth's 3.2 to 3.5 | illustration |
| CO2 190, 280, 560 and 1120 ppm | one step each | 33 to 59 s | instantaneous all-sky change of the net downward flux at the top of the atmosphere against 280 ppm: -1.62, +2.88 and +5.86 W m-2; the second doubling is 1.03 times the first | a number |

**No mountains, the first experiment.** Over hours 12 to 24 the mean absolute
3-hour change of surface pressure is 116.6 Pa against Earth's 105.7 Pa, a
ratio of 1.10, under the 2.0 the design set for a start that is not too
rough. With the uniform offset removed (about -215 m at 500 hPa, the air-mass
rule's), the root-mean-square 500 hPa height difference from Earth over 20 to
80 N is 28 m at day 3 and 50 m at day 5, and its ratio to the perturbed
control's falls from 17 at day 3 to 8.9 at day 5, 5.1 at hour 180 and 2.5 at
day 10. So a no-mountains difference far from the ranges is above the noise
floor for about the first seven and a half days and not after. The 2 m
temperature is up to about 20 K warmer over the lowered plateaus at day 3,
which is mostly the 6.5 K per km the start gives the lowered ground, not
weather.

The 12-hour day ran with `radiation_interval_s = 1200` (the preflight
recommends 600 to keep Earth's 5 degrees of sun per call) and its Earth
control with the shipped 3000, so the two differ by more than the day.

These are runs of days over a sea that does not respond. None of them is
that world's climate.

---

## Named on everything

A world's tapes carry its name, its label for that lead time, its identity,
the hash of its world statement, its settings, whether its coast is its own,
the one sentence, and the sun's true position, as global attributes beginning
`ARWEN_WORLD`. The world's name is also in the tape's title. An Earth tape is
the same bytes as before, with or without an all-Earth table
(`tests/test_arwen_global_world_tape_labels.py`).

---

## Refused on a world

These doors compare against Earth's observations or Earth's geography, and
refuse a run on any other world by name, before any work:

- the regional parent exports, because the regional model takes its terrain,
  coast and land use from Earth's geography;
- cycling, and every leg of the assimilation door;
- the scorecards.

`tests/test_arwen_global_world_refusals.py` holds each door to its reason's
text.

Not supported yet, and why:

| Asked for | Why not |
|---|---|
| any world's own climate, an ice age, a snowball | no ocean or sea ice answers the atmosphere |
| a tidally locked planet | needs a sea that answers its sky |
| another gravity, radius or air composition | the physics kernels and the grid hold Earth's gravity, radius and air |
| more or less air | the radiation refuses layers deeper than 115,000 Pa and looks every layer past its tables' 109,663 Pa up at 109,663 Pa; both carried convection schemes hold live thresholds in absolute pressure (six in Grell-Freitas, seven in New Tiedtke, none in the microphysics) and the vertical coordinate is checked only down to a 500 hPa surface, read row by row in [the thin-air kernel audit](arwen-global-worlds-pressure-audit.md) |
| another year length | ozone, vegetation, the land-use season and the trace gases follow a 365-day calendar |
| another star | the solar spectrum is fixed |
| a regional zoom on a world with its own coast | the regional model reads Earth's geography |

---

## Credits

- **PALEOMAP.** The invented-world maps are Scotese, C. R. and Wright, N. M.
  (2018), *PALEOMAP Paleodigital Elevation Models (PaleoDEMS) for the
  Phanerozoic*, Zenodo, https://doi.org/10.5281/zenodo.5460860, licensed
  Creative Commons Attribution 4.0 International. This package does not
  carry them: the reader fetches them on request and writes this credit into
  every world folder built from them, from where the world statement prints
  it. A world's surface and terrain are a modified form of the map (averaged
  to the model grid, filtered, flooded from the largest ocean), and the
  licence asks anyone who shares that form, or pictures made from it, to
  credit the creators, link the licence, and say that it was modified.
- **Aquaplanet sea temperatures.** The four profiles are those of Neale, R. B.
  and Hoskins, B. J., *A standard test for AGCMs including their physical
  parametrizations: I: The proposal*, Atmospheric Science Letters 1, 101-107,
  doi:10.1006/asle.2000.0019 (cited as 2000 on the Aqua-Planet Experiment's
  sea-surface temperature sheet; the article itself is dated 2001). The
  formulas in the table above are the ones on that sheet, and the package
  implements them from it.

The literature on mountain-removal experiments and on Pangaean climates is not
cited here, because none of it has been checked against a run of this model.
A result on this page that names a paper names one that was read.
