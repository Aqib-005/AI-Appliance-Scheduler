<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Schedule extends Model
{
    use HasFactory;

    protected $fillable = [
        'day',
        'appliance_id',
        'start_hour',
        'end_hour',
    ];

    // Relationship to appliance
    public function appliance()
    {
        return $this->belongsTo(Appliance::class);
    }
}