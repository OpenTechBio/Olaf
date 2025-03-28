import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import {
  FormBuilder,
  FormGroup,
  FormsModule,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import { NavigationExtras, Router } from '@angular/router';
import { Subscription } from 'rxjs';

import { ProjectService } from '../../services/project.service';
import { UserService } from '../../services/user.service';

import { Project, ProjectAgent, ProjectLanguage, ProjectModel } from '../../models/project';

import { HlmButtonDirective } from '@spartan-ng/ui-button-helm';
import {
  BrnDialogCloseDirective,
  BrnDialogContentDirective,
  BrnDialogTriggerDirective,
} from '@spartan-ng/ui-dialog-brain';
import {
  HlmDialogComponent,
  HlmDialogContentComponent,
  HlmDialogFooterComponent,
  HlmDialogHeaderComponent,
  HlmDialogTitleDirective,
} from '@spartan-ng/ui-dialog-helm';
import { HlmIconComponent, provideIcons } from '@spartan-ng/ui-icon-helm';
import { HlmInputDirective } from '@spartan-ng/ui-input-helm';
import { BrnSelectImports } from '@spartan-ng/ui-select-brain';
import { HlmSelectImports } from '@spartan-ng/ui-select-helm';

import { lucidePlus } from '@ng-icons/lucide';

@Component({
  selector: 'app-projects',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    ReactiveFormsModule,

    HlmButtonDirective,

    BrnDialogContentDirective,
    BrnDialogCloseDirective,
    BrnDialogTriggerDirective,
    HlmDialogComponent,
    HlmDialogContentComponent,
    HlmDialogFooterComponent,
    HlmDialogHeaderComponent,
    HlmDialogTitleDirective,

    HlmIconComponent,
    HlmInputDirective,

    BrnSelectImports,
    HlmSelectImports,
  ],
  providers: [provideIcons({ lucidePlus })],
  templateUrl: './projects.component.html',
  styleUrl: './projects.component.scss',
})
export class ProjectsComponent {
  projectSubscription?: Subscription;
  pageNumberSubscription?: Subscription;
  totalPagesSubscription?: Subscription;
  createProjectForm: FormGroup;

  projects?: Project[];
  pageNumber = 1;
  totalPages = 1;

  constructor(
    private formBuilder: FormBuilder,
    public router: Router,
    public projectService: ProjectService,
    public userService: UserService,
  ) {
    this.createProjectForm = this.formBuilder.group({
      name: ['', [Validators.required]],
      language: ['', [Validators.required]],
      model: ['', [Validators.required]],
      agent: ['', [Validators.required]],
    });
  }

  ngOnInit() {
    this.projectService.setPageNumber(1);
    this.projectService.setSearchFilter('');

    this.projectSubscription = this.projectService
      .getProjects()
      .subscribe((projects) => {
        this.projects = projects;
      });
    this.pageNumberSubscription = this.projectService
      .getPageNumber()
      .subscribe((number) => {
        this.pageNumber = number;
      });
    this.totalPagesSubscription = this.projectService
      .getTotalPages()
      .subscribe((total) => {
        this.totalPages = total;
      });
  }

  ngOnDestroy() {
    this.projectSubscription?.unsubscribe();
    this.pageNumberSubscription?.unsubscribe();
    this.totalPagesSubscription?.unsubscribe();
  }

  async createProject() {
    await this.projectService.createProject(
      this.createProjectForm.value.name,
      this.createProjectForm.value.language as ProjectLanguage,
      this.createProjectForm.value.model as ProjectModel,
      this.createProjectForm.value.agent as ProjectAgent,
    );
    console.log(this.createProjectForm.value.agent);
    
    this.createProjectForm.reset();
  }

  goToWorkspace(project: Project) {
    const navigationExtras: NavigationExtras = {
      state: { project: project },
    };
    this.router.navigate(['workspace'], navigationExtras);
  }
}
