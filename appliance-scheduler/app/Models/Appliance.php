<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Factories\HasFactory;

class Appliance extends Model
{
    use HasFactory;

    /**
     * The attributes that are mass assignable.
     *
     * @var array
     */
    protected $fillable = [
        'name',
        'power',
        'preferred_start',
        'preferred_end',
        'duration',
        'usage_days', // Add this
    ];

    public function schedules()
    {
        return $this->hasMany(Schedule::class);
    }
}
