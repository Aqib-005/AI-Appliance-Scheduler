<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class SelectedAppliance extends Model
{
    use HasFactory;

    protected $fillable = [
        'appliance_id',
        'name',
        'power',
        'preferred_start',
        'preferred_end',
        'duration',
        'usage_days',
        'predicted_start_time',
        'predicted_end_time',
    ];

    // relationship with the Appliance model
    public function appliance()
    {
        return $this->belongsTo(Appliance::class, 'appliance_id');
    }
}