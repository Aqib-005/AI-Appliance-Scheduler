<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    /**
     * Run the migrations.
     */
    public function up()
    {
        Schema::table('appliances', function (Blueprint $table) {
            // Change preferred_start and preferred_end to time
            $table->time('preferred_start')->change();
            $table->time('preferred_end')->change();

            // Change duration to decimal
            $table->decimal('duration', 8, 2)->change(); // 8 digits total, 2 decimal places
        });
    }

    public function down()
    {
        Schema::table('appliances', function (Blueprint $table) {
            // Revert changes if needed
            $table->integer('preferred_start')->change();
            $table->integer('preferred_end')->change();
            $table->integer('duration')->change();
        });
    }
};
