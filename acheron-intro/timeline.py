"""Single source of truth for every timecode in the ACHERON intro.

All times are seconds from the first frame. The audio builder and the frame
renderer both read from here, so moving a cue moves picture and sound together.
"""

FPS = 24
DUR = 45.0

# Output canvas. 9:16 vertical UHD. Design coordinates are 1080x1920 and get
# multiplied by the render scale, so 4K is scale 2.0.
DESIGN_W, DESIGN_H = 1080, 1920

# (start, text, optional phoneme override). Start is where the first syllable lands.
VO = [
    (1.0, "They told you to be realistic."),
    (4.0, "To want less."),
    (6.6, "So you went where the light doesn't reach."),
    (10.4, "Nobody."),
    (11.3, "Watches."),
    (12.1, "You."),
    (13.4, "Five in the morning."),
    (15.5, "No audience."),
    (17.6, "No applause."),
    (19.7, "One more rep."),
    (21.7, "Every one of us is crossing something."),
    (24.8, "Most turn back."),
    (26.6, "You didn't."),
    (29.3, "Persistence."),
    (30.5, "Consistency."),
    (31.8, "Loyalty."),
    (33.7, "For the ones still crossing."),
    (37.3, "Acheron.", "ˈækəɹɑːn."),  # AK-uh-ron, not AY-kron
    (38.7, "Ahead of time."),
    (42.0, "The roll is opening."),
    (43.5, "New Year's."),
]

# Heartbeat. Starts at 52 bpm and tightens so the last three beats land on
# NOBODY / WATCHES / YOU.
HEARTBEATS = [6.20, 7.35, 8.50, 9.55, 10.40, 11.30, 12.10]
DUB_OFFSET = 0.27            # second sound of each lub-dub
FINAL_BEAT = 44.15

# Scene 2
LINE_CUT = 6.0               # light line cuts down the centre
RACK_OPEN = 9.0              # line opens into the rack
WORDS = [(10.4, "NOBODY."), (11.3, "WATCHES."), (12.1, "YOU.")]

# Scene 3: one iron-plate clang per cut
CLANGS = [13.4, 15.5, 17.6, 19.7, 21.7]
CAPTIONS = ["5 AM.", "NO AUDIENCE.", "NO APPLAUSE.", "ONE MORE REP.", None]
DRONE_STEPS = [(17.6, 1), (21.7, 2)]      # (time, semitones above root)
BREATHS = [(14.55, "out"), (16.75, "in"), (18.85, "out"), (20.85, "in"), (23.05, "out")]

# Scene 4
WATER_IN = 24.0
SILENCE = (27.55, 28.05)     # everything drops out
BELL = 28.05                 # dagger locks, bell, red ring
VIRTUES = [(29.3, "PERSISTENCE."), (30.5, "CONSISTENCY."), (31.8, "LOYALTY.")]

# Scene 5
SHOCKWAVE = 33.05
LETTERS_START = 33.85
LETTER_STEP = 0.40
LETTER_BURN = 0.55
SWELL_HIT = 36.8             # full wordmark, sub-bass hit
SWEEP = (37.0, 38.5)
TAGLINE_IN = 38.45

# Scene 6
LOCKUP_RISE = (41.0, 42.3)
INVITE = [(42.0, "THE ROLL IS OPENING."), (43.5, "NEW YEAR'S.")]
CUT_TO_BLACK = 44.6
