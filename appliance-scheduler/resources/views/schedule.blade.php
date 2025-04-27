<!DOCTYPE html>
<html>

<head>
    <meta name="csrf-token" content="{{ csrf_token() }}">
    <title>Schedule Appliances</title>
    <link href="{{ asset('css/app.css') }}" rel="stylesheet">
</head>

<body class="schedule-page">

    <header class="app-header">
        <div class="app-header-container">
            <a href="{{ route('dashboard') }}" class="app-header-brand">
                <img src="{{ asset('images/logo.png') }}" alt="App Logo" class="app-header-logo">
                <span class="app-header-title">HomeSched</span>
            </a>

            <h1 class="app-page-title">Schedule Appliances</h1>
        </div>
    </header>

    <div class="container">
        <!-- Left column: appliances + schedule -->
        <div class="appliance-list">
            <h2>Appliances</h2>
            <div class="appliance-list-items">
                @foreach($appliances as $appliance)
                    <button onclick="openSchedulePopup('{{ $appliance->id }}')">
                        {{ $appliance->name }}
                    </button>
                @endforeach
            </div>
            <button class="btn btn-success schedule-button" onclick="runSchedulingAlgorithm()">
                Schedule
            </button>
        </div>

        <!-- Right column: week grid -->
        <div class="weekly-grid">
            @foreach(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'] as $day)
                <div class="day-column" id="day-{{ strtolower($day) }}" onclick="selectDay('{{ $day }}')">
                    <h3>{{ $day }}</h3>
                    <div id="appliances-{{ strtolower($day) }}" class="appliances-container">
                        @foreach($selectedAppliances as $appl)
                            @if(strtolower($appl->usage_days) === strtolower($day))
                                <div class="appliance-item" id="appliance-{{ $appl->id }}">
                                    <div class="action-icons">
                                        <button class="icon-btn" onclick="openEditAppliancePopup('{{ $appl->id }}','{{ $day }}')"
                                            title="Edit">&#9998;</button>
                                        <button class="icon-btn"
                                            onclick="openRemoveConfirmation('{{ $appl->id }}','{{ strtolower($day) }}')"
                                            title="Remove">&#128465;</button>
                                    </div>
                                    <span>
                                        {{ $appl->name }}
                                        ({{ $appl->preferred_start }} - {{ $appl->preferred_end }}, {{ $appl->duration }}h)
                                    </span>
                                </div>
                            @endif
                        @endforeach
                    </div>
                </div>
            @endforeach
        </div>
    </div>

    <!-- Schedule Popup -->
    <div id="schedulePopup" class="popup">
        <h2>Schedule Appliance for <span id="selectedDayName"></span></h2>
        <form id="scheduleForm">
            <input type="hidden" id="applianceId" name="applianceId">
            <label for="applianceName">Appliance:</label>
            <input type="text" id="applianceName" name="applianceName" readonly>
            <br><br>
            <label for="preferredStart">Preferred Start Time:</label>
            <input type="time" id="preferredStart" name="preferredStart" required>
            <br><br>
            <label for="preferredEnd">Preferred End Time:</label>
            <input type="time" id="preferredEnd" name="preferredEnd" required>
            <br><br>
            <label for="duration">Duration (hours):</label>
            <input type="number" id="duration" name="duration" step="0.01" placeholder="e.g., 3.23" required>
            <br><br>
            <button type="button" onclick="addApplianceToDay()">Add</button>
            <button type="button" onclick="closeSchedulePopup()">Cancel</button>
        </form>
    </div>

    <!-- Edit Appliance Popup -->
    <div id="editAppliancePopup" class="popup">
        <h2>Edit Appliance for <span id="editSelectedDayName"></span></h2>
        <form onsubmit="event.preventDefault();">
            <input type="hidden" id="editApplianceId" name="applianceId">
            <label for="editApplianceName">Appliance:</label>
            <input type="text" id="editApplianceName" name="applianceName" readonly>
            <br><br>
            <label for="editPreferredStart">Preferred Start Time:</label>
            <input type="time" id="editPreferredStart" name="preferredStart" required>
            <br><br>
            <label for="editPreferredEnd">Preferred End Time:</label>
            <input type="time" id="editPreferredEnd" name="preferredEnd" required>
            <br><br>
            <label for="editDuration">Duration (hours):</label>
            <input type="number" id="editDuration" name="duration" step="0.01" placeholder="e.g., 2.5" required>
            <br><br>
            <button type="button" class="btn btn-primary" onclick="updateAppliance()">Save</button>
            <button type="button" class="btn" onclick="closeEditAppliancePopup()">Cancel</button>
        </form>
    </div>

    <!-- Remove Confirmation Popup -->
    <div id="removeConfirmationPopup" class="popup">
        <h2>Confirm Removal</h2>
        <p>Are you sure you want to remove this appliance?</p>
        <button type="button" id="confirmRemoveBtn">Confirm</button>
        <button type="button" onclick="closeRemoveConfirmation()">Cancel</button>
    </div>

    <!-- Overlay -->
    <div id="overlay" class="overlay"></div>

    <script>
        let selectedDay = null;
        let selectedApplianceId = null;

        function selectDay(day) {
            document.querySelectorAll('.day-column').forEach(column => {
                column.classList.remove('active');
            });
            document.getElementById(`day-${day.toLowerCase()}`).classList.add('active');
            selectedDay = day;
        }

        function openSchedulePopup(applianceId, applianceName, preferredStart, preferredEnd, duration) {
            if (!selectedDay) {
                alert('Please select a day first.');
                return;
            }
            selectedApplianceId = applianceId;

            fetch(`/appliance/get/${applianceId}`)
                .then(response => response.json())
                .then(data => {
                    const preferredStartFormatted = data.preferred_start ? data.preferred_start.substring(0, 5) : '';
                    const preferredEndFormatted = data.preferred_end ? data.preferred_end.substring(0, 5) : '';

                    document.getElementById('applianceId').value = data.id;
                    document.getElementById('applianceName').value = data.name;
                    document.getElementById('preferredStart').value = preferredStartFormatted;
                    document.getElementById('preferredEnd').value = preferredEndFormatted;
                    document.getElementById('duration').value = data.duration;
                    document.getElementById('selectedDayName').textContent = selectedDay || '';

                    document.getElementById('schedulePopup').classList.add('active');
                    document.getElementById('overlay').classList.add('active');
                })
                .catch(error => {
                    console.error('Error fetching appliance data:', error);
                    alert('Failed to fetch appliance data. Please try again.');
                });
        }

        function closeSchedulePopup() {
            document.getElementById('schedulePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        function addApplianceToDay() {
            const start = document.getElementById('preferredStart').value;
            const end = document.getElementById('preferredEnd').value;
            const dur = parseFloat(document.getElementById('duration').value);

            if (!start || !end || isNaN(dur) || dur <= 0) {
                alert('Please fill in all fields correctly.');
                return;
            }
            if (end <= start) {
                alert('End time must be later than start time on the same day.');
                return;
            }
            const [sh, sm] = start.split(':').map(Number);
            const [eh, em] = end.split(':').map(Number);
            const availableHours = ((eh * 60 + em) - (sh * 60 + sm)) / 60;
            if (availableHours < dur) {
                alert(`The selected window (${availableHours.toFixed(2)}h) is shorter than the required duration of ${dur.toFixed(2)}h.`);
                return;
            }

            fetch('/selected-appliance/add', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                },
                body: JSON.stringify({
                    appliance_id: selectedApplianceId,
                    name: document.getElementById('applianceName').value,
                    preferred_start: start,
                    preferred_end: end,
                    duration: dur,
                    usage_days: selectedDay
                })
            })
                .then(r => r.json())
                .then(data => {
                    if (!data.success) throw new Error();
                    const item = document.createElement('div');
                    item.className = 'appliance-item';
                    item.id = `appliance-${data.selected_appliance_id}`;
                    item.innerHTML = `
                  <div class="action-icons">
                    <button class="icon-btn"
                        onclick="openEditAppliancePopup('${data.selected_appliance_id}','${selectedDay}')"
                        title="Edit">&#9998;</button>
                    <button class="icon-btn"
                        onclick="openRemoveConfirmation('${data.selected_appliance_id}','${selectedDay.toLowerCase()}')"
                        title="Remove">&#128465;</button>
                  </div>
                  <span>
                    ${document.getElementById('applianceName').value}
                    (${start} - ${end}, ${dur.toFixed(2)}h)
                  </span>`;
                    document
                        .getElementById(`appliances-${selectedDay.toLowerCase()}`)
                        .appendChild(item);
                    closeSchedulePopup();
                })
                .catch(() => alert('Failed to add appliance.'));
        }

        function openEditAppliancePopup(applianceId, day) {
            fetch(`/selected-appliance/get/${applianceId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('editApplianceId').value = data.appliance.id;
                        document.getElementById('editApplianceName').value = data.appliance.name;
                        document.getElementById('editPreferredStart').value = data.appliance.preferred_start;
                        document.getElementById('editPreferredEnd').value = data.appliance.preferred_end;
                        document.getElementById('editDuration').value = data.appliance.duration;
                        document.getElementById('editSelectedDayName').textContent = day;

                        document.getElementById('editAppliancePopup').classList.add('active');
                        document.getElementById('overlay').classList.add('active');
                    } else {
                        alert('Failed to load appliance details.');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    alert('An error occurred while loading appliance details.');
                });
        }

        function updateAppliance() {
            let start = document.getElementById('editPreferredStart').value;
            let end = document.getElementById('editPreferredEnd').value;
            let dur = parseFloat(document.getElementById('editDuration').value);
            const id = document.getElementById('editApplianceId').value;

            start = start.substring(0, 5);
            end = end.substring(0, 5);

            // basic completeness
            if (!start || !end || isNaN(dur) || dur <= 0) {
                alert('Please fill in all fields correctly.');
                return;
            }
            // ensure ordering
            if (end <= start) {
                alert('End time must be later than start time on the same day.');
                return;
            }
            // ensure window ≥ duration
            const [sh, sm] = start.split(':').map(Number);
            const [eh, em] = end.split(':').map(Number);
            const availableHours = ((eh * 60 + em) - (sh * 60 + sm)) / 60;
            if (availableHours < dur) {
                alert(`The window (${availableHours.toFixed(2)}h) is shorter than the required duration (${dur.toFixed(2)}h).`);
                return;
            }

            fetch(`/selected-appliance/update/${id}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': document.querySelector('meta[name="csrf-token"]').content
                },
                body: JSON.stringify({
                    preferred_start: start,
                    preferred_end: end,
                    duration: dur
                })
            })
                .then(r => {
                    if (!r.ok) return r.text().then(t => Promise.reject(t));
                    return r.json();
                })
                .then(data => {
                    if (!data.success) throw new Error();
                    const el = document.getElementById(`appliance-${id}`);
                    el.querySelector('span').textContent =
                        `${document.getElementById('editApplianceName').value}
                     (${start} - ${end}, ${dur.toFixed(2)}h)`;
                    closeEditAppliancePopup();
                })
                .catch(err => {
                    console.error(err);
                    alert('Failed to update appliance.');
                });
        }

        function closeEditAppliancePopup() {
            document.getElementById('editAppliancePopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }

        function openRemoveConfirmation(id, day) {
            pendingRemoveId = id;
            pendingRemoveDay = day;
            document.getElementById('removeConfirmationPopup').classList.add('active');
            document.getElementById('overlay').classList.add('active');
        }
        function closeRemoveConfirmation() {
            document.getElementById('removeConfirmationPopup').classList.remove('active');
            document.getElementById('overlay').classList.remove('active');
        }
        document.getElementById('confirmRemoveBtn').addEventListener('click', () => {
            fetch(`/selected-appliance/remove/${pendingRemoveId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}'
                }
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById(`appliance-${pendingRemoveId}`).remove();
                    } else {
                        alert('Remove failed');
                    }
                })
                .finally(closeRemoveConfirmation);
        });

        function runSchedulingAlgorithm() {
            fetch('/schedule', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRF-TOKEN': '{{ csrf_token() }}',
                },
                body: JSON.stringify({}),
            })
                .then(response => {
                    if (!response.ok) {
                        return response.text().then(text => {
                            throw new Error('Network response was not ok');
                        });
                    }
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        window.location.href = data.redirect_url;
                    } else {
                        alert('Failed to generate schedule. Please try again.');
                    }
                })
                .catch(error => {
                    console.error('API Error:', error);
                    alert('Failed to generate schedule. Please try again.');
                });
        }
    </script>
</body>

</html>