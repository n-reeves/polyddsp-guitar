def output_to_notes_polyphonic(
    frames: torch.Tensor,
    onsets: torch.Tensor,
    contours: torch.Tensor,
    onset_thresh: float = 0.5,
    frame_thresh: float = 0.3,
    min_note_len: int = 10,
    infer_onsets: bool = True,
    max_freq: Optional[float] = None,
    min_freq: Optional[float] = None,
    melodia_trick: bool = True,
    n_voices: int = 10,
    energy_tol: int = 11,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert pitch predictions to note predictions

    Args:
        frames: The frame predictions (freq_bins, time_frames)
        onsets: The onset predictions (freq_bins, time_frames)
        onset_thresh: The threshold for onset detection
        frame_thresh: The threshold for frame detection
        min_note_len: The minimum number of frames for a note to be valid
        infer_onsets: If True, infer onsets from large changes in frame amplitude
        max_freq: The maximum frequency to allow
        min_freq: The minimum frequency to allow
        melodia_trick: If True, use the Melodia trick to remove spurious notes
        energy_tol: The energy tolerance for the Melodia trick

    Returns:
        a tuple containing the notes tensor (n_voices, time_frames) and velocity tensor (n_voices, time_frames)
    """
    n_frames = frames.shape[-1]
    n_batch = frames.shape[0]

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    # use onsets inferred from frames in addition to the predicted onsets
    if infer_onsets:
        onsets = get_infered_onsets(onsets, frames)

    peak_thresh_mat = torch.zeros_like(onsets)
    peaks = argrelmax(onsets)
    peak_thresh_mat[peaks] = onsets[peaks]

    # permute to make time dimension 1, to ensure time is sorted before frequency
    onset_idx = torch.nonzero(peak_thresh_mat.permute([0, 2, 1]) >= onset_thresh)
    # return columns to original order
    onset_idx = torch.cat(
        [onset_idx[:, 0:1], onset_idx[:, 2:3], onset_idx[:, 1:2]], dim=1
    )
    # sort backwards in time?
    onset_idx = onset_idx.flip([0])

    remaining_energy = torch.clone(frames)

    notes = torch.zeros((n_batch, n_voices, n_frames), device=onsets.device)
    amplitude = torch.zeros((n_batch, n_voices, n_frames), device=onsets.device)

    # from each onset_idx, search for strings of frames that are above the frame threshold in remaining_energy, allowing for gaps shorter than energy_tol
    for batch_idx, freq_idx, note_start_idx in onset_idx:
        # if we're too close to the end of the audio, continue
        if note_start_idx >= n_frames - 1:
            continue

        # find time index at this frequency band where the frames drop below an energy threshold
        i = note_start_idx + 1
        k = 0  # number of frames since energy dropped below threshold
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[batch_idx, freq_idx, i] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1

        i -= k  # go back to frame above threshold

        # if the note is too short, skip it
        if i - note_start_idx <= min_note_len:
            continue

        remaining_energy[batch_idx, freq_idx, note_start_idx:i] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[batch_idx, freq_idx + 1, note_start_idx:i] = 0
        if freq_idx > 0:
            remaining_energy[batch_idx, freq_idx - 1, note_start_idx:i] = 0

        bends = get_pitch_bends(
            contours, [batch_idx, freq_idx + MIDI_OFFSET, note_start_idx, i], 25
        )

        # need to assign notes to voices, first in first out.
        # keep track of voice allocation order
        v = list(range(n_voices))
        for j in range(n_voices):
            if notes[batch_idx, v[j], note_start_idx] == 0:
                notes[batch_idx, v[j], note_start_idx:i] = utils.tensor_midi_to_hz(
                    bends
                )
                amplitude[batch_idx, v[j], note_start_idx:i] = frames[
                    batch_idx, freq_idx, note_start_idx:i
                ]
                v.insert(0, v.pop(j))
                break
            # if no free voice set the lowest amplitude voice to the new note
            if j == n_voices - 1:
                min_idx = torch.argmin(
                    torch.mean(amplitude[batch_idx, :, note_start_idx:i], dim=1)
                )
                notes[batch_idx, min_idx, note_start_idx:i] = utils.tensor_midi_to_hz(
                    bends
                )
                amplitude[batch_idx, min_idx, note_start_idx:i] = frames[
                    batch_idx, freq_idx, note_start_idx:i
                ]
                v.insert(0, v.pop(v.index(min_idx)))

    if melodia_trick:
        energy_shape = remaining_energy.shape

        while torch.max(remaining_energy) > frame_thresh:
            batch, freq_idx, i_mid = utils.unravel_index(
                torch.argmax(remaining_energy), energy_shape
            )
            remaining_energy[batch, freq_idx, i_mid] = 0

            # forward pass
            i = i_mid + 1
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[batch, freq_idx, i] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[batch, freq_idx, i] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[batch, freq_idx + 1, i] = 0
                if freq_idx > 0:
                    remaining_energy[batch, freq_idx - 1, i] = 0

                i += 1

            i_end = i - 1 - k  # go back to frame above threshold

            # backward pass
            i = i_mid - 1
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[batch, freq_idx, i] < frame_thresh:
                    k += 1
                else:
                    k = 0

                remaining_energy[batch, freq_idx, i] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[batch, freq_idx + 1, i] = 0
                if freq_idx > 0:
                    remaining_energy[batch, freq_idx - 1, i] = 0

                i -= 1

            i_start = i + 1 + k  # go back to frame above threshold
            assert i_start >= 0, "{}".format(i_start)
            assert i_end < n_frames

            if i_end - i_start <= min_note_len:
                # note is too short, skip it
                continue

            bends = get_pitch_bends(
                contours, [batch, freq_idx + MIDI_OFFSET, i_start, i_end], 25
            )

            # if there is a gap in available voices, add the note
            v = list(range(n_voices))
            for j in range(n_voices):
                if notes[batch, v[j], i_start:i_end].sum == 0:
                    notes[batch, v[j], i_start:i_end] = utils.tensor_midi_to_hz(bends)
                    amplitude[batch, v[j], i_start:i_end] = frames[
                        batch, freq_idx, i_start:i_end
                    ]

    return notes, amplitude