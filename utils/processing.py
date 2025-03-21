import numpy as np
import argparse
import csv
import os
import time
import logging
import h5py
import librosa
import subprocess
import shutil

from .utilities import (
    create_folder, float32_to_int16, create_logging,
    get_filename, read_metadata, read_midi, read_maps_midi
)

from . import config

def pack_maestro_dataset_to_hdf5(args):
    """Load & resample MAESTRO audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate

    # Paths
    csv_path = os.path.join(dataset_dir, 'maestro-v3.0.0.csv')
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maestro')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # Read meta dict
    meta_dict = read_metadata(csv_path)

    audios_num = len(meta_dict['canonical_composer'])
    logging.info('Total audios number: {}'.format(audios_num))

    feature_time = time.time()

    # Load & resample each audio file to a hdf5 file
    for n in range(audios_num):
        logging.info('{} {}'.format(n, meta_dict['midi_filename'][n]))

        # Read midi
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][n])
        midi_dict = read_midi(midi_path)

        # Load audio
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n])
        (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(
            os.path.splitext(meta_dict['audio_filename'][n])[0]))

        create_folder(os.path.dirname(packed_hdf5_path))

        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('canonical_composer', data=meta_dict['canonical_composer'][n].encode(), dtype='S100')
            hf.attrs.create('canonical_title', data=meta_dict['canonical_title'][n].encode(), dtype='S100')
            hf.attrs.create('split', data=meta_dict['split'][n].encode(), dtype='S20')
            hf.attrs.create('year', data=meta_dict['year'][n].encode(), dtype='S10')
            hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=meta_dict['audio_filename'][n].encode(), dtype='S100')
            hf.attrs.create('duration', data=meta_dict['duration'][n], dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        
    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


def pack_maps_dataset_to_hdf5(args):
    """MAPS is a piano dataset only used for evaluating our piano transcription
    system (optional).

    Load & resample MAPS audio files, then write to hdf5 files.

    Args:
      dataset_dir: str, directory of dataset
      workspace: str, directory of your workspace
    """
    dataset_dir = args.dataset_dir
    workspace = args.workspace

    sample_rate = config.sample_rate
    pianos = ['ENSTDkCl', 'ENSTDkAm']

    # Paths
    waveform_hdf5s_dir = os.path.join(workspace, 'hdf5s', 'maps')

    logs_dir = os.path.join(workspace, 'logs', get_filename(__file__))
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    feature_time = time.time()
    count = 0

    # Load & resample each audio file to a hdf5 file
    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, 'MUS')

        audio_names = [os.path.splitext(name)[0] for name in os.listdir(sub_dir) 
            if os.path.splitext(name)[-1] == '.mid']
        
        for audio_name in audio_names:
            print('{} {}'.format(count, audio_name))
            audio_path = '{}.wav'.format(os.path.join(sub_dir, audio_name))
            midi_path = '{}.mid'.format(os.path.join(sub_dir, audio_name))

            (audio, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            midi_dict = read_maps_midi(midi_path)
            
            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, '{}.h5'.format(audio_name))
            create_folder(os.path.dirname(packed_hdf5_path))

            with h5py.File(packed_hdf5_path, 'w') as hf:
                hf.attrs.create('split', data='test'.encode(), dtype='S20')
                hf.attrs.create('midi_filename', data='{}.mid'.format(audio_name).encode(), dtype='S100')
                hf.attrs.create('audio_filename', data='{}.wav'.format(audio_name).encode(), dtype='S100')
                hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
                hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
            
            count += 1

    logging.info('Write hdf5 to {}'.format(packed_hdf5_path))
    logging.info('Time: {:.3f} s'.format(time.time() - feature_time))


def convert_and_copy_midi_files(args):
    """
    Walks through every MIDI file in the dataset directory, converts it to a corresponding WAV file using fluidsynth,
    and copies both the MIDI file and the generated WAV file to the workspace directory, preserving the original directory structure.
    """
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    # Define the soundfont path used by fluidsynth
    soundfont = "/Users/user/.soundfonts/Dore Mark's Yamaha S6-v1.6.sf2"

    # Set up basic logging if not already configured
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith('.midi'):
                midi_path = os.path.join(root, file)
                # Compute the relative directory from dataset_dir
                rel_dir = os.path.relpath(root, dataset_dir)
                dest_dir = os.path.join(workspace, rel_dir)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Copy the MIDI file to the destination directory
                dest_midi_path = os.path.join(dest_dir, file)
                shutil.copy2(midi_path, dest_midi_path)
                logging.info("Copied MIDI: {}".format(dest_midi_path))
                
                # Define destination WAV file path (same base name but with .wav extension)
                base_name = os.path.splitext(file)[0]
                dest_wav_path = os.path.join(dest_dir, base_name + '.wav')
                
                # Build the fluidsynth command
                cmd = [
                    "fluidsynth", "-ni", soundfont,
                    midi_path, "-F", dest_wav_path,
                    "-r", "44100", "-g", "2"
                ]
                
                try:
                    subprocess.run(cmd, check=True)
                    logging.info("Converted to WAV: {}".format(dest_wav_path))
                except subprocess.CalledProcessError as e:
                    logging.error("Error converting {}: {}".format(midi_path, e))
    
    logging.info("Conversion and copying completed in {:.3f} seconds.".format(time.time() - start_time))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process and convert piano dataset files.')
    subparsers = parser.add_subparsers(dest='mode', required=True)

    # Existing subparsers for packing datasets to HDF5
    parser_pack_maestro = subparsers.add_parser('pack_maestro_dataset_to_hdf5')
    parser_pack_maestro.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maestro.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')

    parser_pack_maps = subparsers.add_parser('pack_maps_dataset_to_hdf5')
    parser_pack_maps.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_maps.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    
    # New subparser for the convert and copy functionality
    parser_convert = subparsers.add_parser('convert_and_copy')
    parser_convert.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_convert.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    
    # Parse arguments and call the corresponding function
    args = parser.parse_args()
    
    if args.mode == 'pack_maestro_dataset_to_hdf5':
        pack_maestro_dataset_to_hdf5(args)
        
    elif args.mode == 'pack_maps_dataset_to_hdf5':
        pack_maps_dataset_to_hdf5(args)
        
    elif args.mode == 'convert_and_copy':
        convert_and_copy_midi_files(args)
        
    else:
        raise Exception('Incorrect arguments!')

