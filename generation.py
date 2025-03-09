import pretty_midi #to parse midi files to strings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pygame

class MelodyGen:
    # constructor, include the path to the MIDI file, an array for the c major notes, 
    # and the decision tree with max depth of 5 to avoid overfitting
    def __init__(self, MIDI_path):
        self.MIDI_path = MIDI_path
        self.c_maj_notes = []
        self.tree = DecisionTreeClassifier(max_depth=5)
    
    #extract the notes from MIDI file using pretty_midi
    def extract_notes(self, midi_file):
        notes_lst = []

        #create PrettyMIDI obj
        data = pretty_midi.PrettyMIDI(midi_file)   

        #loop through the instruments in the MIDI file and extract the pitch of all the notes
        for ins in data.instruments:
            for note in ins.notes:
                notes_lst.append(note.pitch)
        #print(notes)
        return notes_lst

    #converts the notes list to c major scale 
    def c_major_scale(self, notes_lst):
        c_major = {60, 62,64,65,67,69,71}
        self.c_maj_notes = [note for note in notes_lst if note in c_major]
        return self.c_maj_notes
    
    #create a mapping of 2 notes to one (seq_len = 2)
    def pairing_notes(self, seq_len = 2):

        #x represents the notes that will be mapped to y
        X, y = [], []

        #loop over the c major scale notes to map them in order
        for i in range(len(self.c_maj_notes) - seq_len):
            X.append((self.c_maj_notes[i:i+seq_len]))
            y.append((self.c_maj_notes[i+seq_len]))

        return np.array(X), np.array(y)
    
    #train decision tree model according to our note mapping
    def train_model(self):
        X, y = self.pairing_notes()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.tree.fit(X_train, y_train)

    #generate notes according to the starting notes and number of notes
    def gen_melody(self, starting_notes, num_notes=10):

        #preprocess the data and train the decision tree
        notes_lst = self.extract_notes(self.MIDI_path)
        self.c_major_scale(notes_lst)
        self.train_model()

        #loop in range of input (to generage num_notes number of notes)
        for _ in range(num_notes):
            #take the last two notes of the starting notes and reshape them to have 1 column
            X_input = np.array(starting_notes[-2:]).reshape(1,-1)

            # create probability distribution for every note after a pair of notes 
            # returns a 2d arr, use the 1st row of probabilities)
            prob_dist = self.tree.predict_proba(X_input)[0]

            #store the list of possible notes to predit as a variable
            possible_notes = self.tree.classes_

            #generate the next node randomly using the probability distribution and possible notes
            next_note = np.random.choice(possible_notes, p=prob_dist)

            starting_notes.append(next_note)
        return starting_notes
    
    def generate_midi(self, generated_notes, output_path="generated_melody.mid"):
        #create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  #piano
        start_time = 0
        
        #use generated notes
        for note in generated_notes:
            end_time = start_time + 0.3  #set duration of each note to 0.5 seconds
            instrument.notes.append(pretty_midi.Note(velocity=100, pitch=note, start=start_time, end=end_time))
            start_time += 0.3  #update start time for the next note

        #append instruments to the object
        midi.instruments.append(instrument)

        #save the midi file
        midi.write(output_path)
        #print(f"file saved as {output_path}")

    def play_midi(self, midi_file):
        #initialize pygame mixer
        pygame.mixer.init()
        
        #load and play the file
        pygame.mixer.music.load(midi_file)
        pygame.mixer.music.play()

        #keep the program running while the music plays
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

#melody = MelodyGen("C:\\users\\owner\\Downloads\\mozart_sym40g_III_com.mid")
melody = MelodyGen("C:\\Users\\Owner\\Downloads\\Bach_inv13a_295.mid")
#melody = MelodyGen("C:\\Users\\Owner\\Downloads\\Handel_HarBlE_595.mid")
#melody = MelodyGen("C:\\Users\\Owner\\Downloads\\Handel_HarBlE_com.mid")
start_notes = [67,69]
generated_notes = melody.gen_melody(start_notes)
melody.generate_midi(generated_notes, "generated_melody.mid")
# Play the generated MIDI file
melody.play_midi("generated_melody.mid")
print('hi')