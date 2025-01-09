<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;

class SelectedAppliance extends Model
{
    use HasFactory;

    protected $fillable = [
        'name',
        'preferred_start',
        'preferred_end',
        'usage_days',
    ];

    protected $casts = [
        'usage_days' => 'array',
    ];
}